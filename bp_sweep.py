import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import csv
import time

# ── EXPERIMENT MATRIX ─────────────────────────────────────────────────────────
# These are the variables you sweep. Every combination gets its own run.
# This is your core run matrix from section 5.4 of the proposal.
DEPTH_LIST      = [2, 4, 6]          # hidden layers — start with 3, add 8 later
TRAIN_SIZES     = [100, 500, 1000, 5000]
SEEDS           = [0, 1, 2]          # minimum 3 for debug runs; 5+ for final
HIDDEN_SIZE     = 256
EPOCHS          = 30
LR              = 1e-3
BATCH_SIZE      = 64
DEVICE          = torch.device("cpu")

# ── DATA ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

# Download once, reuse across all runs.
# Loading inside the sweep loop would re-download every iteration.
full_train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# ── MODEL ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
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

# ── TRAINING AND EVALUATION ───────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimiser):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            correct += (model(inputs).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ── LOGGING ───────────────────────────────────────────────────────────────────
# Every run writes one row per epoch to a CSV.
# This is your experiment log — never delete it, never overwrite it.
# If you lose this file, you lose your results.
# Columns match exactly what section 10 of the proposal says to record.
LOG_FILE = "bp_results.csv"
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'algorithm', 'num_layers', 'hidden_size', 'train_size',
        'seed', 'epoch', 'train_loss', 'test_acc', 'wall_time_s'
    ])

# ── SWEEP ─────────────────────────────────────────────────────────────────────
# Total runs = len(DEPTH_LIST) * len(TRAIN_SIZES) * len(SEEDS)
# = 3 * 4 * 3 = 36 runs for this debug matrix.
# Each run is one (depth, train_size, seed) combination.
total = len(DEPTH_LIST) * len(TRAIN_SIZES) * len(SEEDS)
run = 0

for num_layers in DEPTH_LIST:
    for train_size in TRAIN_SIZES:
        for seed in SEEDS:
            run += 1
            print(f"\nRun {run}/{total} | layers={num_layers} "
                  f"train_size={train_size} seed={seed}")

            # Set seed before anything random — model init, data sampling.
            # Must be set inside the loop, not once at the top, because
            # each run needs its own independent random state.
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Sample training subset.
            # Using the same seed means the same 100/500/1000/5000 examples
            # are selected every time you re-run this script — reproducible.
            indices = torch.randperm(len(full_train))[:train_size]
            train_loader = DataLoader(
                Subset(full_train, indices),
                batch_size=min(BATCH_SIZE, train_size),
                # batch_size must not exceed train_size — with 100 examples
                # and batch_size=64, you get 1-2 batches per epoch which is
                # fine, but batch_size > train_size would give you 0 batches.
                shuffle=True
            )

            model     = MLP(num_layers).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimiser = optim.Adam(model.parameters(), lr=LR)

            start = time.time()
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                for epoch in range(1, EPOCHS + 1):
                    train_loss = train_epoch(model, train_loader,
                                             criterion, optimiser)
                    test_acc   = evaluate(model, test_loader)
                    elapsed    = time.time() - start

                    writer.writerow([
                        'BP', num_layers, HIDDEN_SIZE, train_size,
                        seed, epoch, round(train_loss, 6),
                        round(test_acc, 4), round(elapsed, 2)
                    ])

                    if epoch % 10 == 0:
                        print(f"  Epoch {epoch:2d} | loss: {train_loss:.4f} "
                              f"| acc: {test_acc:.4f} | {elapsed:.1f}s")

print(f"\nDone. Results saved to {LOG_FILE}")