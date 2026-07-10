import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import csv
import time

# ── EXPERIMENT MATRIX ─────────────────────────────────────────────────────────
# Mirrors bp_sweep.py, plus the extra PC-only variable: inference step budget.
# Every combination of (depth, train_size, seed, inference_budget) is one run.
DEPTH_LIST      = [2, 4, 6]              # hidden layers
TRAIN_SIZES     = [100, 500, 1000, 5000]
SEEDS           = [0, 1, 2]
INFERENCE_BUDGETS = [10, 20, 50]        # PC-only sweep variable
HIDDEN_SIZE     = 256
OUTPUT_SIZE     = 10
NUM_LAYERS      = 2                      # placeholder; overridden per run
EPOCHS          = 30
LR_WEIGHTS      = 5e-4
LR_INFERENCE    = 0.1
BATCH_SIZE      = 64
DEVICE          = torch.device("cpu")

# INFERENCE_STEPS is a module-level global that train_pc / evaluate_pc read at
# call time (exactly as in PC.py). We REASSIGN it before each run so the
# algorithm functions — copied verbatim from PC.py — pick up the current
# budget without any modification to their bodies. See sweep loop below.
INFERENCE_STEPS = INFERENCE_BUDGETS[0]

# ── DATA ──────────────────────────────────────────────────────────────────────
# Same MNIST loading + preprocessing as bp_sweep.py (Normalize 0.1307/0.3081,
# flatten to 784). Downloaded once and reused across all runs.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

full_train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Fixed evaluation subset: 1000 test examples sampled ONCE with a dedicated
# seed (999) so the exact same subset is used across all 108 runs. Evaluating
# on this subset instead of the full 10,000-example set cuts per-eval cost 10x
# while keeping the comparison identical across runs. evaluate_pc is unchanged;
# it just receives this smaller DataLoader.
EVAL_SUBSET_SIZE = 1000
EVAL_SEED        = 999
_g = torch.Generator().manual_seed(EVAL_SEED)
eval_indices = torch.randperm(len(test_data), generator=_g)[:EVAL_SUBSET_SIZE]
test_loader  = DataLoader(Subset(test_data, eval_indices),
                          batch_size=256, shuffle=False)

# ── PC NETWORK ────────────────────────────────────────────────────────────────
# Copied verbatim from PC.py. Do not redesign the algorithm here.
class PCNetwork(nn.Module):
    """
    Predictive coding network following Whittington & Bogacz (2017).

    Architecture: input(784) -> hidden x NUM_LAYERS(256) -> output(10)

    Key design: each layer owns TOP-DOWN weights that predict the layer below.
    W[i] maps from layer i+1 activations DOWN to layer i activations.

    Layer sizes: [784, 256, 256, 10] for NUM_LAYERS=2

    The output layer activations ARE the class scores (size 10).
    No separate classification head — the output activations are clamped
    to one-hot targets during training and used directly for classification
    during evaluation (argmax). This removes the untrained-head bug.
    """
    def __init__(self, input_size=784, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super().__init__()

        self.layer_sizes = ([input_size] +
                           [hidden_size] * num_layers +
                           [output_size])
        self.num_layers  = len(self.layer_sizes) - 1
        self.output_size = output_size

        # Top-down weight matrices: self.weights[i] maps FROM layer i+1
        # DOWN TO layer i. So self.weights[i] has shape
        # (layer_sizes[i], layer_sizes[i+1]) as an nn.Linear.
        # nn.Linear(in, out) stores weight as (out, in), so:
        # nn.Linear(layer_sizes[i+1], layer_sizes[i]) correctly maps
        # a vector of size layer_sizes[i+1] to size layer_sizes[i].
        self.weights = nn.ModuleList([
            nn.Linear(self.layer_sizes[i+1], self.layer_sizes[i])
            for i in range(self.num_layers)
        ])

        self.activation = nn.Tanh()

    def predict_layer(self, activations_above, layer_idx):
        """
        Generate top-down prediction of layer layer_idx,
        given activations of layer layer_idx+1.
        prediction = tanh(W[layer_idx] @ activations_above + bias)
        """
        return self.activation(self.weights[layer_idx](activations_above))

    def forward_initialise(self, x):
        """
        Bottom-up pass to initialise activations before inference.
        Uses transpose of top-down weights as an approximation of
        a bottom-up pass. This is standard practice — exact form
        of initialisation matters less than starting in a reasonable range.

        Returns list of activation tensors:
        activations[0] = input (clamped)
        activations[1..N] = hidden layers (free during inference)
        activations[N+1] = output (clamped to label during training,
                                   free during evaluation)
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
        """
        Prediction error at layer i:
            error[i] = activations[i] - predict_layer(activations[i+1], i)
        """
        errors = []
        for i in range(self.num_layers):
            prediction = self.predict_layer(activations[i+1], i)
            errors.append(activations[i] - prediction)
        return errors

    def inference_step(self, activations, lr, clamp_output=True):
        """
        One inference step: update hidden activations to reduce
        total prediction error energy.

            dE/da[i] = error[i]                           (from layer above)
                     - f'(z[i-1]) * W[i-1].weight.t() @ error[i-1]
                                                           (from layer below)

        Update rule: activations[i] -= lr * dE/da[i]

        This matches W&B (2017) equation 10, with the sign correction
        from the March 2025 Rosenbaum errata applied.
        """
        errors = self.compute_errors(activations)
        new_activations = [activations[0]]  # input always clamped

        # Highest layer index. When clamp_output=True (training) we stop before
        # it and keep the clamped one-hot target. When clamp_output=False
        # (evaluation) we let it settle so the settled output IS the prediction.
        top = len(activations) - 1
        last_free = top - 1 if clamp_output else top

        for i in range(1, last_free + 1):
            # Own-error term: only exists if a layer above predicts layer i,
            # i.e. errors[i] is defined (i < num_layers). The top layer has no
            # layer above it, so it has no own-error term — its energy comes
            # solely from how well it predicts the layer below.
            if i < self.num_layers:
                grad = errors[i].clone()
            else:
                grad = torch.zeros_like(activations[i])

            # Below term: how changing activations[i] affects the prediction
            # error at layer i-1. z[i-1] = W[i-1] @ activations[i] + bias;
            # f'(tanh(z)) = 1 - tanh(z)^2. Present for every i >= 1.
            with torch.no_grad():
                z_below    = self.weights[i-1](activations[i])
                f_prime    = 1.0 - self.activation(z_below) ** 2
                # (batch, size[i-1]) @ (size[i-1], size[i]) -> (batch, size[i])
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
        Update weights using settled activations from inference.

        We use autograd to compute dE/dW[i] — the gradient of energy
        w.r.t. weights. Activations are detached so autograd only
        differentiates through the weights, not through the inference
        history (which was computed manually above).
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


# ── TRAINING ──────────────────────────────────────────────────────────────────
# Copied verbatim from PC.py. Per-example inference loop (NOT batched).
# Reads module globals INFERENCE_STEPS and LR_INFERENCE at call time.
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


# ── EVALUATION ────────────────────────────────────────────────────────────────
# Copied verbatim from PC.py. Evaluation via inference with a FREE output layer.
def evaluate_pc(model, loader):
    """
    Evaluation via INFERENCE (the correct PC prediction rule):
    clamp the input, leave the output layer FREE, run the same inference
    dynamics used in training, then argmax the settled output layer.
    """
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for inputs, labels in loader:
            for idx in range(inputs.size(0)):
                x           = inputs[idx].unsqueeze(0)
                activations = model.forward_initialise(x)
                # Output left free; settle via inference, then read it out.
                activations = model.run_inference(
                    activations, INFERENCE_STEPS, LR_INFERENCE,
                    clamp_output=False)
                pred = activations[-1].argmax(dim=1).item()
                if pred == labels[idx].item():
                    correct += 1
                total += 1
    return correct / total


# ── LOGGING ───────────────────────────────────────────────────────────────────
# One row per epoch, written INSIDE the epoch loop so a crash keeps every
# completed epoch. Columns add `inference_steps` relative to bp_results.csv.
LOG_FILE = "pc_results.csv"
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'algorithm', 'num_layers', 'hidden_size', 'train_size',
        'seed', 'inference_steps', 'epoch', 'train_energy',
        'test_acc', 'wall_time_s'
    ])

# ── SWEEP ─────────────────────────────────────────────────────────────────────
# Total runs = 3 depths * 4 train sizes * 3 seeds * 3 inference budgets = 108.
total = (len(DEPTH_LIST) * len(TRAIN_SIZES) *
         len(SEEDS) * len(INFERENCE_BUDGETS))
run = 0

for num_layers in DEPTH_LIST:
    for train_size in TRAIN_SIZES:
        for seed in SEEDS:
            for inference_steps in INFERENCE_BUDGETS:
                run += 1
                print(f"\nRun {run}/{total} | layers={num_layers} "
                      f"train_size={train_size} seed={seed} "
                      f"T={inference_steps}")

                # Reassign the module global read by train_pc / evaluate_pc.
                INFERENCE_STEPS = inference_steps

                # Seed set per RUN (inside the innermost loop) before any
                # randomness — model init and data sampling. Because the seed
                # is reset here, the three inference budgets sharing a given
                # (depth, size, seed) get IDENTICAL initial weights and the
                # IDENTICAL training subset, so inference_steps is a cleanly
                # controlled variable. See interpretation note in the summary.
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Same training subset selection as bp_sweep.py.
                indices = torch.randperm(len(full_train))[:train_size]
                train_loader = DataLoader(
                    Subset(full_train, indices),
                    batch_size=min(BATCH_SIZE, train_size),
                    shuffle=True
                )

                model     = PCNetwork(num_layers=num_layers).to(DEVICE)
                optimiser = optim.Adam(model.weights.parameters(),
                                       lr=LR_WEIGHTS)

                start = time.time()
                last_acc = None  # carried forward on non-evaluation epochs
                with open(LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for epoch in range(1, EPOCHS + 1):
                        train_energy = train_pc(model, train_loader, optimiser)

                        # Evaluate only every 5 epochs (and the final epoch) to
                        # cut eval cost ~5x. On skipped epochs we still write a
                        # row (for continuous energy tracking) and carry the
                        # most recent accuracy forward. `last_acc` is None until
                        # the first evaluation at epoch 5, so epochs 1-4 log ''.
                        if epoch % 5 == 0 or epoch == EPOCHS:
                            last_acc = evaluate_pc(model, test_loader)
                        elapsed  = time.time() - start

                        acc_to_log = round(last_acc, 4) if last_acc is not None else ''
                        writer.writerow([
                            'PC', num_layers, HIDDEN_SIZE, train_size,
                            seed, inference_steps, epoch,
                            round(train_energy, 6),
                            acc_to_log, round(elapsed, 2)
                        ])
                        f.flush()  # ensure each epoch is on disk immediately

                        if epoch % 10 == 0:
                            print(f"  Run {run}/{total} | Epoch {epoch:2d} "
                                  f"| energy: {train_energy:.4f} "
                                  f"| acc: {last_acc:.4f} | {elapsed:.1f}s")

print(f"\nDone. Results saved to {LOG_FILE}")
