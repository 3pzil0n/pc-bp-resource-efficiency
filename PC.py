import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import csv
import time

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
HIDDEN_SIZE     = 256
NUM_LAYERS      = 2
BATCH_SIZE      = 64
EPOCHS          = 20
LR_WEIGHTS      = 5e-4   # lowered from 1e-3: 1e-3 overfit the 1000-example
                         # set, causing accuracy to *fall* after epoch 1
LR_INFERENCE    = 0.1    # raised from 0.05 so inference settles within the
                         # step budget (energy plateaus instead of still
                         # dropping steeply at the last step)
INFERENCE_STEPS = 30     # raised from 20: at 20 steps energy had not yet
                         # settled, so both training targets and eval readout
                         # were taken from an un-relaxed state
TRAIN_SIZE      = 1000
OUTPUT_SIZE     = 10
DEVICE          = torch.device("cpu")
SEED            = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── DATA ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

train_data   = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data    = datasets.MNIST('./data', train=False, download=True, transform=transform)
indices      = torch.randperm(len(train_data))[:TRAIN_SIZE]
train_loader = DataLoader(Subset(train_data, indices), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=256, shuffle=False)

# ── PC NETWORK ────────────────────────────────────────────────────────────────
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
            # self.weights[i].weight has shape (layer_sizes[i], layer_sizes[i+1])
            # To go bottom-up: multiply current (shape: batch x layer_sizes[i])
            # by weight.t() (shape: layer_sizes[i+1] x layer_sizes[i]).T
            # = weight (shape: layer_sizes[i] x layer_sizes[i+1]).T
            # Wait — nn.Linear stores weight as (out_features, in_features)
            # so self.weights[i].weight shape = (layer_sizes[i], layer_sizes[i+1])
            # Bottom-up: (batch x layer_sizes[i]) @ (layer_sizes[i] x layer_sizes[i+1])
            # = (batch x layer_sizes[i+1]) ✓
            current = self.activation(
                current @ self.weights[i].weight
            )
            activations.append(current.detach().clone())
        return activations

    def compute_errors(self, activations):
        """
        Prediction error at layer i:
            error[i] = activations[i] - predict_layer(activations[i+1], i)
        
        = actual activations at layer i MINUS what layer i+1 predicts
          layer i should be doing.
        
        Positive error: actual > predicted (layer above under-predicts you)
        Negative error: actual < predicted (layer above over-predicts you)
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
        
        For hidden layer i (not input, not output), the gradient of
        total energy w.r.t. activations[i] is:
        
            dE/da[i] = error[i]                           (from layer above)
                     - f'(z[i-1]) * W[i-1].weight.t() @ error[i-1]
                                                           (from layer below)
        
        Where:
        - error[i] = activations[i] - f(W[i] @ activations[i+1])
          This is how wrong the layer above's prediction of us is.
          Gradient w.r.t. activations[i] is simply +error[i].
        
        - The second term is how our activations affect the prediction
          error at the layer below. W[i-1] generates a prediction of
          layer i-1 FROM layer i. Changing activations[i] changes that
          prediction, which changes error[i-1]. Chain rule gives:
          d(error[i-1])/d(activations[i]) = -f'(z[i-1]) * W[i-1].weight
          So contribution to dE/da[i] = -W[i-1].weight.t() @ 
                                         (f'(z[i-1]) * error[i-1])
        
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
        
        Energy at layer i: E[i] = 0.5 * ||error[i]||^2
        Total energy: E = sum_i E[i]
        
        We use autograd to compute dE/dW[i] — the gradient of energy
        w.r.t. weights. Activations are detached so autograd only
        differentiates through the weights, not through the inference
        history (which was computed manually above).
        
        This gives the same update as the local Hebbian rule in W&B:
        dW[i] proportional to error[i] outer_product activations[i+1]
        but computed via autograd for correctness and simplicity.
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
def diagnose_inference(model, loader, n=5):
    """
    DIAGNOSTIC: confirm the inference loop actually reduces prediction-error
    energy. For a handful of training examples we clamp input + one-hot output
    (exactly as during training) and print the total energy at each inference
    step. A correct implementation shows energy decreasing monotonically toward
    a plateau; a flat or rising curve means the inference gradient/lr is wrong.
    """
    model.eval()
    inputs, labels = next(iter(loader))
    print("\n[diagnostic] total energy per inference step (train-mode clamping)")
    print(f"{'ex':>3} | " + " ".join(f"t{t:02d}" for t in
          [0, 1, 2, 5, 10, INFERENCE_STEPS]))
    with torch.no_grad():
        for idx in range(min(n, inputs.size(0))):
            x      = inputs[idx].unsqueeze(0)
            target = torch.zeros(1, model.output_size)
            target[0, labels[idx].item()] = 1.0

            activations    = model.forward_initialise(x)
            activations[-1] = target

            energies = [model.total_energy(activations)]
            for _ in range(INFERENCE_STEPS):
                activations = model.inference_step(activations, LR_INFERENCE)
                energies.append(model.total_energy(activations))

            sample = [energies[t] for t in [0, 1, 2, 5, 10, INFERENCE_STEPS]]
            print(f"{idx:>3} | " + " ".join(f"{e:6.2f}" for e in sample))
    print("[diagnostic] energy should fall monotonically ↓\n")


def evaluate_pc(model, loader):
    """
    Evaluation via INFERENCE (the correct PC prediction rule):

    Clamp the input, leave the output layer FREE, and run the same inference
    dynamics used in training. The network settles hidden AND output activations
    to the state that best explains the input under the learned generative
    model. The settled output layer is then the class score — argmax gives the
    prediction.

    This replaces the previous forward-pass-only evaluation, which read out the
    transpose of the top-down weights. Those weights are trained as a top-down
    generator, not a bottom-up classifier, so the forward pass was near-random —
    the cause of the chance-level oscillation despite falling energy.
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


# ── RUN ───────────────────────────────────────────────────────────────────────
model     = PCNetwork().to(DEVICE)
optimiser = optim.Adam(model.weights.parameters(), lr=LR_WEIGHTS)

print(f"Training PC | layers={NUM_LAYERS} | hidden={HIDDEN_SIZE} | "
      f"train_size={TRAIN_SIZE} | T={INFERENCE_STEPS}")
print("-" * 60)

# Sanity-check the inference dynamics before training so we know the loop
# reduces energy (independent of whether the weights have learned anything).
diagnose_inference(model, train_loader)

start = time.time()
for epoch in range(1, EPOCHS + 1):
    energy   = train_pc(model, loader=train_loader, optimiser=optimiser)
    test_acc = evaluate_pc(model, loader=test_loader)
    elapsed  = time.time() - start
    print(f"Epoch {epoch:2d} | energy: {energy:.4f} | "
          f"test acc: {test_acc:.4f} | time: {elapsed:.1f}s")
    if epoch == 1:
        diagnose_inference(model, train_loader)  # re-check after learning starts