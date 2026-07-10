# Resource Efficiency of Predictive Coding vs. Backpropagation

Code accompanying the paper *"Resource Efficiency of Predictive Coding versus
Backpropagation in Matched Nonlinear Classification Networks: A Sample–Compute
Trade-off Analysis."*

The experiments compare backpropagation (BP) and a Whittington–Bogacz-style
supervised predictive-coding (PC) network on MNIST, characterising the joint
**sample–compute trade-off** across network depth, labelled training-set size,
and PC inference budget.

## Repository contents

| File | Description |
|------|-------------|
| `PC.py` | Core predictive-coding implementation (network, iterative inference step, weight update, train/eval loops) following Whittington & Bogacz (2017). Imported by the confirmatory sweep. |
| `bp_sweep.py` | Backpropagation baseline sweep over depths `{2, 4, 6}`, training sizes `{100, 500, 1000, 5000}`, and seeds `{0, 1, 2}`. |
| `pc_sweep.py` | Exploratory PC sweep. Mirrors `bp_sweep.py` and adds the PC-only inference-budget variable `T ∈ {10, 20, 50}`. |
| `confirmatory_sweep.py` | Confirmatory protocol implementing all **six comparability controls** (shared fixed test subset, training-only timing, matched initial weights, seeded shuffling, tuned PC learning rates, compute-proxy logging). This is the script behind the main-paper results. |

Each sweep writes a CSV of per-epoch results (accuracy, wall-clock training
time, optimiser steps, and — for PC — total inference steps).

## Requirements

- Python 3.14
- PyTorch
- torchvision
- numpy

Install with:

```bash
pip install torch torchvision numpy
```

The experiments in the paper were run on Apple Silicon CPU
(`device = torch.device("cpu")`). MNIST is downloaded automatically by
torchvision on first run.

## Usage

Run the confirmatory sweep (the protocol used for the main-paper results):

```bash
python confirmatory_sweep.py
```

Run the exploratory baselines / pilot sweeps:

```bash
python bp_sweep.py
python pc_sweep.py
```

Hyperparameters (depths, training sizes, seeds, inference budgets, learning
rates, epochs) are set as constants near the top of each script; edit them
there to reproduce or extend the reported configurations. The confirmatory
sweep is compute-intensive because PC performs per-example iterative inference —
expect it to run for an extended period on CPU.

## Reference

Whittington, J. C. R., & Bogacz, R. (2017). An Approximation of the Error
Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian
Synaptic Plasticity. *Neural Computation*, 29(5), 1229–1262.
