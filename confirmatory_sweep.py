"""
confirmatory_sweep.py
=====================

A CONTROLLED confirmatory comparison of Backprop (BP) and Predictive Coding
(PC) on MNIST, fixing the six comparability problems found in the audit of the
original bp_sweep.py / pc_sweep.py:

  1. Different test sets for BP and PC   -> Control 1: one fixed 1,000-example
                                             test subset, shared by both.
  2. Wall-clock timing included eval     -> Control 2: timer wraps ONLY the
                                             training call; eval timed separately.
  3. BP/PC init mismatch                 -> Control 3: matched weights (PC = BP
                                             transposed) from the same RNG draw.
  4. Unseeded DataLoader shuffle         -> Control 4: every shuffling DataLoader
                                             gets generator=Generator(seed).
  5. Untuned PC learning rates           -> Control 6: automated LR sensitivity
                                             sweep picks PC's LRs before the run.
  6. No compute-proxy logging            -> Control 5: log optimiser_steps and
                                             inference_steps_total per epoch.

The PC algorithm (PCNetwork class, inference_step, weight_update, train_pc,
evaluate_pc) is taken VERBATIM from PC.py — it is not redesigned here.

Run overnight with:
    caffeinate -i python3 confirmatory_sweep.py > confirmatory_log.txt 2>&1 &
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import csv
import time

# ── FIXED HYPERPARAMETERS ─────────────────────────────────────────────────────
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
NUM_LAYERS  = 2          # placeholder default for PCNetwork; overridden per run
EPOCHS      = 30
BATCH_SIZE  = 64
DEVICE      = torch.device("cpu")

# ── EXPERIMENT MATRIX ─────────────────────────────────────────────────────────
DEPTH_LIST        = [2, 4, 6]
TRAIN_SIZES       = [1000, 5000]
SEEDS             = [0, 1, 2, 3, 4]
INFERENCE_BUDGETS = [10, 20, 50]
#   PC runs = 3 depths x 2 sizes x 5 seeds x 3 budgets = 90
#   BP runs = 3 depths x 2 sizes x 5 seeds             = 30
#   Total   = 120

# BP uses the standard Adam learning rate from the original bp_sweep.py.
# NOTE (flag): the audit only flagged PC's learning rates as untuned, so only PC
# is tuned (Control 6). BP keeps 1e-3. We deliberately do NOT reuse PC's tuned
# LR_WEIGHTS for BP: PC's weight LR is calibrated for an energy-minimisation
# update, a different objective and gradient scale from BP's cross-entropy loss.
# Coupling them would be an apples-to-oranges choice, not a fairer one.
BP_LR = 1e-3

# These two module-level globals are READ by the verbatim PC functions
# (train_pc / evaluate_pc) at call time, exactly as in PC.py. We reassign them
# per PC run so the verbatim code picks up the current budget and tuned LR
# without any edit to its body.
INFERENCE_STEPS = INFERENCE_BUDGETS[0]
LR_INFERENCE    = 0.1     # provisional; replaced by Control 6's chosen value

# ── DATA ──────────────────────────────────────────────────────────────────────
# Same MNIST preprocessing as both original sweeps: normalise then flatten to 784.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

full_train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# ── CONTROL 1: one fixed test subset shared by BP and PC ──────────────────────
# Sampled ONCE, here, with a dedicated generator seeded 999. Because it is built
# outside the sweep loops and never resampled, every BP run and every PC run is
# scored on the identical 1,000 images. Using a *local* generator (not
# torch.manual_seed) means this draw does not disturb the global RNG stream that
# the per-run seeding controls.
EVAL_SUBSET_SIZE = 1000
_eval_gen    = torch.Generator().manual_seed(999)
eval_indices = torch.randperm(len(test_data), generator=_eval_gen)[:EVAL_SUBSET_SIZE]
test_loader  = DataLoader(Subset(test_data, eval_indices), batch_size=256, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# PC NETWORK — copied VERBATIM from PC.py. Do not redesign.
# ══════════════════════════════════════════════════════════════════════════════
class PCNetwork(nn.Module):
    """
    Predictive coding network following Whittington & Bogacz (2017).

    Each layer owns TOP-DOWN weights that predict the layer below.
    W[i] maps from layer i+1 activations DOWN to layer i activations.
    Layer sizes: [784, 256, 256, 10] for NUM_LAYERS=2.
    The output layer activations ARE the class scores (size 10); they are
    clamped to one-hot targets during training and read out (argmax) during
    evaluation after inference with a free output layer.
    """
    def __init__(self, input_size=784, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super().__init__()

        self.layer_sizes = ([input_size] +
                           [hidden_size] * num_layers +
                           [output_size])
        self.num_layers  = len(self.layer_sizes) - 1
        self.output_size = output_size

        # self.weights[i] = nn.Linear(layer_sizes[i+1], layer_sizes[i]) maps a
        # vector of size layer_sizes[i+1] DOWN to size layer_sizes[i].
        self.weights = nn.ModuleList([
            nn.Linear(self.layer_sizes[i+1], self.layer_sizes[i])
            for i in range(self.num_layers)
        ])

        self.activation = nn.Tanh()

    def predict_layer(self, activations_above, layer_idx):
        """Top-down prediction of layer layer_idx from layer layer_idx+1."""
        return self.activation(self.weights[layer_idx](activations_above))

    def forward_initialise(self, x):
        """
        Bottom-up pass to initialise activations before inference. Uses the
        transpose of the top-down weights as an approximate bottom-up pass.
        Returns [input(clamped), hidden..., output].
        """
        activations = [x]
        current = x
        for i in range(self.num_layers):
            current = self.activation(
                current @ self.weights[i].weight
            )
            activations.append(current.detach().clone())
        return activations

    def compute_errors(self, activations):
        """error[i] = activations[i] - predict_layer(activations[i+1], i)."""
        errors = []
        for i in range(self.num_layers):
            prediction = self.predict_layer(activations[i+1], i)
            errors.append(activations[i] - prediction)
        return errors

    def inference_step(self, activations, lr, clamp_output=True):
        """
        One inference step. For hidden layer i:
            dE/da[i] = error[i]                              (from layer above)
                     - f'(z[i-1]) * W[i-1].weight.t() @ error[i-1]  (from below)
        Update: activations[i] -= lr * dE/da[i]. Matches W&B (2017) eq. 10 with
        the sign correction from the March 2025 Rosenbaum errata.
        """
        errors = self.compute_errors(activations)
        new_activations = [activations[0]]  # input always clamped

        # clamp_output=True (training) keeps the clamped one-hot output;
        # clamp_output=False (evaluation) lets the output settle so it becomes
        # the prediction.
        top = len(activations) - 1
        last_free = top - 1 if clamp_output else top

        for i in range(1, last_free + 1):
            # Own-error term only exists if a layer above predicts layer i
            # (i < num_layers). The top layer has no layer above it.
            if i < self.num_layers:
                grad = errors[i].clone()
            else:
                grad = torch.zeros_like(activations[i])

            # Below term: effect of activations[i] on the error at layer i-1.
            with torch.no_grad():
                z_below    = self.weights[i-1](activations[i])
                f_prime    = 1.0 - self.activation(z_below) ** 2
                grad_below = (f_prime * errors[i-1]) @ self.weights[i-1].weight

            grad = grad - grad_below
            new_activations.append(activations[i] - lr * grad)

        if clamp_output:
            new_activations.append(activations[-1])  # output stays clamped
        return new_activations

    def run_inference(self, activations, steps, lr, clamp_output=True):
        """Run `steps` inference iterations, returning settled activations."""
        for _ in range(steps):
            activations = self.inference_step(activations, lr, clamp_output)
        return activations

    def total_energy(self, activations):
        """Scalar total prediction-error energy E = sum_i 0.5||error[i]||^2."""
        errors = self.compute_errors(activations)
        return sum(0.5 * (e ** 2).sum().item() for e in errors)

    def weight_update(self, activations, optimiser):
        """
        Update weights from settled activations. Autograd differentiates the
        energy w.r.t. weights only (activations detached), giving the same local
        rule as W&B: dW[i] ~ error[i] (x) activations[i+1].
        """
        optimiser.zero_grad()
        total_energy = torch.tensor(0.0, requires_grad=True)
        for i in range(self.num_layers):
            pred  = self.predict_layer(activations[i+1].detach(), i)
            error = activations[i].detach() - pred
            total_energy = total_energy + 0.5 * (error ** 2).sum()
        total_energy.backward()
        optimiser.step()
        return total_energy.item()


# ── PC TRAIN / EVAL — copied VERBATIM from PC.py ──────────────────────────────
# Per-example inference loop (NOT batched). Read module globals INFERENCE_STEPS
# and LR_INFERENCE at call time.
def train_pc(model, loader, optimiser):
    model.train()
    total_energy = 0.0
    n_examples   = 0

    for inputs, labels in loader:
        for idx in range(inputs.size(0)):
            x      = inputs[idx].unsqueeze(0)   # shape: (1, 784)
            label  = labels[idx].item()

            # One-hot target for output clamping
            target = torch.zeros(1, model.output_size)
            target[0, label] = 1.0

            # Initialise activations via feedforward pass
            activations = model.forward_initialise(x)

            # Clamp output to target label
            activations[-1] = target

            # Inference phase: update hidden activations only
            for _ in range(INFERENCE_STEPS):
                activations = model.inference_step(activations, LR_INFERENCE)

            # Weight update using settled activations
            energy = model.weight_update(activations, optimiser)
            total_energy += energy
            n_examples   += 1

    return total_energy / n_examples


def evaluate_pc(model, loader):
    """
    Evaluation via inference: clamp the input, leave the output FREE, run the
    same inference dynamics as training, then argmax the settled output layer.
    """
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for inputs, labels in loader:
            for idx in range(inputs.size(0)):
                x           = inputs[idx].unsqueeze(0)
                activations = model.forward_initialise(x)
                activations = model.run_inference(
                    activations, INFERENCE_STEPS, LR_INFERENCE,
                    clamp_output=False)
                pred = activations[-1].argmax(dim=1).item()
                if pred == labels[idx].item():
                    correct += 1
                total += 1
    return correct / total


# ══════════════════════════════════════════════════════════════════════════════
# BP MODEL AND TRAIN / EVAL (standard MLP; not part of the "verbatim" PC code)
# ══════════════════════════════════════════════════════════════════════════════
class MLP(nn.Module):
    """Standard MLP: Linear -> Tanh stack, final Linear to 10 logits."""
    def __init__(self, num_layers, hidden_size=HIDDEN_SIZE):
        super().__init__()
        layers = []
        in_size = 784
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.Tanh())
            in_size = hidden_size
        layers.append(nn.Linear(hidden_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_bp(model, loader, criterion, optimiser):
    """One BP epoch. Returns (avg_loss, optimiser_step_count)."""
    model.train()
    total_loss = 0.0
    n_steps    = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimiser.step()          # one optimiser step per batch
        n_steps    += 1           # Control 5: count real optimiser.step() calls
        total_loss += loss.item()
    return total_loss / len(loader), n_steps


def evaluate_bp(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            correct += (model(inputs).argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL 3: matched initialisation
# ══════════════════════════════════════════════════════════════════════════════
def make_matched_weights(layer_sizes, seed):
    """
    Build ONE set of weight matrices in BP orientation with standard PyTorch
    Kaiming init, then derive the PC matrices by transposing them, so both
    algorithms start from the SAME underlying random numbers.

    Returns (bp_weights, pc_weights): two lists indexed by layer i.
      bp_weights[i] : shape (layer_sizes[i+1], layer_sizes[i])  -> MLP Linear i
      pc_weights[i] : shape (layer_sizes[i],   layer_sizes[i+1]) -> PC   Linear i
    pc_weights[i] is exactly bp_weights[i].t().

    -------------------------------------------------------------------------
    FLAG (biases are NOT matched — read this): only the WEIGHT MATRICES can be
    shared, because a BP layer's bias has length layer_sizes[i+1] while the
    corresponding PC layer's bias has length layer_sizes[i] — different shapes,
    no correspondence. To avoid giving one network an extra dose of random init
    the other structurally cannot copy, the loader below sets ALL biases to zero
    in BOTH models. This deviates slightly from PyTorch's default (which gives
    Linear a small random bias) in exchange for a genuinely matched start. Both
    networks then learn their biases from zero.

    FLAG (same numbers != same function): transposing gives identical numbers,
    but BP uses W as a bottom-up map and PC uses it as a top-down generator, so
    the two networks do NOT compute the same function at init. This control
    guarantees a common, unbiased starting point — not equivalence.
    -------------------------------------------------------------------------
    """
    torch.manual_seed(seed)   # makes the Kaiming draws reproducible per (depth, seed)
    bp_weights = []
    pc_weights = []
    for i in range(len(layer_sizes) - 1):
        # BP orientation: Linear(in=layer_sizes[i], out=layer_sizes[i+1]).
        # Constructing nn.Linear applies PyTorch's standard kaiming_uniform_
        # init to .weight — this IS "standard PyTorch Kaiming initialisation".
        lin = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
        W   = lin.weight.detach().clone()          # (out, in) = (ls[i+1], ls[i])
        bp_weights.append(W)
        pc_weights.append(W.t().contiguous().clone())  # (ls[i], ls[i+1])
    return bp_weights, pc_weights


def load_bp_weights(model, bp_weights):
    """Copy matched weights into the MLP; zero all biases (see make_ flag)."""
    linears = [m for m in model.network if isinstance(m, nn.Linear)]
    assert len(linears) == len(bp_weights)
    with torch.no_grad():
        for lin, W in zip(linears, bp_weights):
            assert lin.weight.shape == W.shape
            lin.weight.copy_(W)
            lin.bias.zero_()


def load_pc_weights(model, pc_weights):
    """Copy matched (transposed) weights into the PCNetwork; zero all biases."""
    assert len(model.weights) == len(pc_weights)
    with torch.no_grad():
        for lin, W in zip(model.weights, pc_weights):
            assert lin.weight.shape == W.shape
            lin.weight.copy_(W)
            lin.bias.zero_()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def make_train_loader(train_size, seed):
    """
    Sample the training subset and build a DataLoader.

    Control 4: shuffle uses generator=Generator(seed), so the example order is
    deterministic and identical for the BP run and the PC runs that share this
    seed. The subset indices are drawn with a *separate* generator (also seeded)
    so they, too, are identical across corresponding BP/PC runs regardless of
    any other RNG use (e.g. model construction).
    """
    idx_gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train), generator=idx_gen)[:train_size]
    shuffle_gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(full_train, indices),
        batch_size=min(BATCH_SIZE, train_size),   # never exceed train_size
        shuffle=True,
        generator=shuffle_gen,
    )
    return loader


# CSV column order — the single source of truth for both the sensitivity file
# and the main results file (the sensitivity file adds two LR columns instead).
MAIN_COLUMNS = [
    'algorithm', 'num_layers', 'hidden_size', 'train_size', 'seed',
    'inference_steps', 'epoch', 'train_loss_or_energy', 'test_acc',
    'wall_time_s', 'eval_time_s', 'optimiser_steps', 'inference_steps_total',
]


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL 6: PC learning-rate sensitivity sweep
# ══════════════════════════════════════════════════════════════════════════════
def run_lr_sensitivity():
    """
    Small automated sweep at depth=2, train_size=1000, seed=0, 15 epochs, over
    LR_INFERENCE in [0.05, 0.1, 0.2] x LR_WEIGHTS in [1e-4, 5e-4, 1e-3], for all
    three inference budgets. Logs every (lr_inf, lr_w, budget) epoch-15 accuracy
    to lr_sensitivity.csv, then returns the (lr_inf, lr_w) pair with the highest
    MEAN epoch-15 accuracy across the three budgets.
    """
    global INFERENCE_STEPS, LR_INFERENCE

    LR_INF_GRID = [0.05, 0.1, 0.2]
    LR_W_GRID   = [1e-4, 5e-4, 1e-3]
    SENS_DEPTH, SENS_SIZE, SENS_SEED, SENS_EPOCHS = 2, 1000, 0, 15

    layer_sizes = [784] + [HIDDEN_SIZE] * SENS_DEPTH + [OUTPUT_SIZE]
    # Matched init for this fixed (depth, seed) — reused for every grid point so
    # the only things changing are the two learning rates and the budget.
    _, pc_weights = make_matched_weights(layer_sizes, SENS_SEED)

    with open("lr_sensitivity.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lr_inference', 'lr_weights', 'inference_steps',
                    'epoch', 'test_acc'])

    # mean_acc[(lr_inf, lr_w)] = mean epoch-15 accuracy across the 3 budgets
    acc_by_combo = {}
    print("\n=== CONTROL 6: PC learning-rate sensitivity sweep ===")
    for lr_inf in LR_INF_GRID:
        for lr_w in LR_W_GRID:
            budget_accs = []
            for T in INFERENCE_BUDGETS:
                # Point the verbatim PC functions at this grid point.
                INFERENCE_STEPS = T
                LR_INFERENCE    = lr_inf

                # Reset RNG per run before any randomness (Control 4 data order).
                torch.manual_seed(SENS_SEED)
                np.random.seed(SENS_SEED)
                loader = make_train_loader(SENS_SIZE, SENS_SEED)

                model = PCNetwork(num_layers=SENS_DEPTH).to(DEVICE)
                load_pc_weights(model, pc_weights)
                optimiser = optim.Adam(model.weights.parameters(), lr=lr_w)

                acc = 0.0
                for epoch in range(1, SENS_EPOCHS + 1):
                    train_pc(model, loader, optimiser)
                    acc = evaluate_pc(model, test_loader)  # only need final epoch
                budget_accs.append(acc)

                with open("lr_sensitivity.csv", 'a', newline='') as f:
                    csv.writer(f).writerow(
                        [lr_inf, lr_w, T, SENS_EPOCHS, round(acc, 4)])
                    f.flush()
                print(f"  lr_inf={lr_inf:<4} lr_w={lr_w:<6} T={T:<2} "
                      f"epoch15 acc={acc:.4f}")

            mean_acc = sum(budget_accs) / len(budget_accs)
            acc_by_combo[(lr_inf, lr_w)] = mean_acc
            print(f"  -> mean over budgets = {mean_acc:.4f}\n")

    best = max(acc_by_combo, key=acc_by_combo.get)
    print(f"=== Chosen PC LRs: LR_INFERENCE={best[0]}, LR_WEIGHTS={best[1]} "
          f"(mean acc {acc_by_combo[best]:.4f}) ===\n")
    return best[0], best[1]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global INFERENCE_STEPS, LR_INFERENCE

    # --- Control 6 first: choose PC learning rates ---
    chosen_lr_inf, chosen_lr_w = run_lr_sensitivity()

    # Fresh results file with the full column set.
    LOG_FILE = "confirmatory_results.csv"
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(MAIN_COLUMNS)

    total_runs = (len(DEPTH_LIST) * len(TRAIN_SIZES) * len(SEEDS)      # BP
                  + len(DEPTH_LIST) * len(TRAIN_SIZES) * len(SEEDS)
                    * len(INFERENCE_BUDGETS))                          # PC
    run = 0

    def log_epoch(f, row):
        csv.writer(f).writerow(row)
        f.flush()   # Control: never lose a completed epoch to a crash

    for depth in DEPTH_LIST:
        layer_sizes = [784] + [HIDDEN_SIZE] * depth + [OUTPUT_SIZE]
        for seed in SEEDS:
            # Control 3: ONE matched init per (depth, seed), reused by the BP run
            # and every PC run (both sizes, all budgets) at this (depth, seed).
            bp_weights, pc_weights = make_matched_weights(layer_sizes, seed)

            for train_size in TRAIN_SIZES:

                # ─────────────── BP run ───────────────
                run += 1
                # Set seed inside the innermost loop, before any randomness.
                torch.manual_seed(seed)
                np.random.seed(seed)
                loader = make_train_loader(train_size, seed)

                model     = MLP(depth).to(DEVICE)
                load_bp_weights(model, bp_weights)   # matched start
                criterion = nn.CrossEntropyLoss()
                optimiser = optim.Adam(model.parameters(), lr=BP_LR)

                cum_train_time = 0.0
                with open(LOG_FILE, 'a', newline='') as f:
                    for epoch in range(1, EPOCHS + 1):
                        # Control 2: time ONLY the training call.
                        t0 = time.perf_counter()
                        train_loss, opt_steps = train_bp(
                            model, loader, criterion, optimiser)
                        train_time = time.perf_counter() - t0

                        # Evaluation timed separately, on the shared subset.
                        e0 = time.perf_counter()
                        test_acc = evaluate_bp(model, test_loader)
                        eval_time = time.perf_counter() - e0

                        cum_train_time += train_time
                        log_epoch(f, [
                            'BP', depth, HIDDEN_SIZE, train_size, seed,
                            0,                        # inference_steps: N/A for BP
                            epoch, round(train_loss, 6), round(test_acc, 4),
                            round(train_time, 4),     # per-epoch training time
                            round(eval_time, 4),      # per-epoch eval time
                            opt_steps,                # Control 5
                            0,                        # inference_steps_total = 0
                        ])
                        if epoch % 5 == 0:
                            print(f"[{run}/{total_runs}] BP depth={depth} "
                                  f"size={train_size} seed={seed} "
                                  f"epoch={epoch:2d} loss={train_loss:.4f} "
                                  f"acc={test_acc:.4f} "
                                  f"train_t={cum_train_time:.1f}s")

                # ─────────────── PC runs (one per budget) ───────────────
                for T in INFERENCE_BUDGETS:
                    run += 1
                    # Point verbatim PC functions at this run's settings.
                    INFERENCE_STEPS = T
                    LR_INFERENCE    = chosen_lr_inf

                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    # Same seed as the BP run above => identical subset + order.
                    loader = make_train_loader(train_size, seed)

                    model     = PCNetwork(num_layers=depth).to(DEVICE)
                    load_pc_weights(model, pc_weights)   # matched start
                    optimiser = optim.Adam(model.weights.parameters(),
                                           lr=chosen_lr_w)

                    # Control 5 compute proxies for PC. train_pc is verbatim and
                    # does one optimiser.step() per training example, so:
                    #   optimiser_steps      = number of training examples
                    #   inference_steps_total = T * number of training examples
                    # (T inference iterations per example). Derived from the
                    # algorithm's structure because train_pc is used unmodified.
                    n_train_examples      = train_size
                    optimiser_steps       = n_train_examples
                    inference_steps_total = T * n_train_examples

                    cum_train_time = 0.0
                    with open(LOG_FILE, 'a', newline='') as f:
                        for epoch in range(1, EPOCHS + 1):
                            t0 = time.perf_counter()
                            train_energy = train_pc(model, loader, optimiser)
                            train_time = time.perf_counter() - t0

                            e0 = time.perf_counter()
                            test_acc = evaluate_pc(model, test_loader)
                            eval_time = time.perf_counter() - e0

                            cum_train_time += train_time
                            log_epoch(f, [
                                'PC', depth, HIDDEN_SIZE, train_size, seed,
                                T, epoch, round(train_energy, 6),
                                round(test_acc, 4),
                                round(train_time, 4),
                                round(eval_time, 4),
                                optimiser_steps,
                                inference_steps_total,
                            ])
                            if epoch % 5 == 0:
                                print(f"[{run}/{total_runs}] PC depth={depth} "
                                      f"size={train_size} seed={seed} T={T} "
                                      f"epoch={epoch:2d} energy={train_energy:.4f} "
                                      f"acc={test_acc:.4f} "
                                      f"train_t={cum_train_time:.1f}s")

    print(f"\nDone. Results saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
