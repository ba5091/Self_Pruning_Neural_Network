"""
main.py — Self-pruning MLP on CIFAR-10 with neuron-level gating.

λ calibration for neuron-level L1 on gates
────────────────────────────────────────────
sparsity_loss = λ × mean(gates) ≈ λ × 0.88 at init
CE loss at init ≈ 2.3  (ln 10)

λ=0.5 → penalty ≈ 0.44  (~19% of CE) → mild,   high accuracy
λ=1.5 → penalty ≈ 1.32  (~57% of CE) → medium
λ=4.0 → penalty ≈ 3.52  (>CE)        → aggressive, needs warmup

Training schedule (25 epochs total):
  Epochs  0–4  : warmup — λ_eff=0, pure classification
  Epochs  5–9  : ramp   — λ_eff linearly 0 → λ
  Epochs 10–24 : full   — λ_eff=λ with cosine LR decay
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from src.model import PrunableNN
from src.train import train, evaluate
from src.utils import (calculate_sparsity, neuron_sparsity,
                       plot_gates, print_layer_stats, print_summary)

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_EPOCHS = 25
WARMUP_END   = 5       # epochs 0-4: no sparsity loss
RAMP_END     = 10      # epochs 5-9: λ linearly ramps 0 → λ_target
WEIGHT_LR    = 1e-3
GATE_LR      = 5e-3    # gates learn 5× faster than weights

# Calibrated for gate-mean L1 (see module docstring)
LAMBDAS = [0.5, 1.5, 4.0]

print(f"Device : {DEVICE}")
print(f"Epochs : {TOTAL_EPOCHS}  "
      f"(warmup={WARMUP_END}, ramp={RAMP_END-WARMUP_END}, "
      f"full={TOTAL_EPOCHS-RAMP_END})")
print(f"Lambdas: {LAMBDAS}\n")

# ── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_data = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
test_data  = datasets.CIFAR10('./data', train=False,               transform=transform)

train_data = Subset(train_data, range(5000))
test_data  = Subset(test_data,  range(1000))

train_loader = DataLoader(train_data, batch_size=128, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False, num_workers=0)

# ── Lambda sweep ─────────────────────────────────────────────────────────────
results = []

for lam in LAMBDAS:
    print(f"\n{'=' * 65}")
    print(f"  Training with λ = {lam}")
    print(f"{'=' * 65}")

    model     = PrunableNN().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameter_groups(weight_lr=WEIGHT_LR, gate_lr=GATE_LR),
        weight_decay=0.0,    # all regularisation comes from sparsity_loss
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-5
    )

    header = (f"  {'Ep':>3}  {'Phase':>5}  {'λ_eff':>6}  "
              f"{'Loss':>7}  {'CE':>7}  {'SpLoss':>7}  "
              f"{'Acc':>7}  {'W-Spar':>8}  {'N-Spar':>8}")
    print(header)
    print(f"  {'-' * 68}")

    for epoch in range(TOTAL_EPOCHS):

        # λ schedule
        if epoch < WARMUP_END:
            current_lam = 0.0
            phase = "warm"
        elif epoch < RAMP_END:
            progress    = (epoch - WARMUP_END + 1) / (RAMP_END - WARMUP_END)
            current_lam = lam * progress
            phase = "ramp"
        else:
            current_lam = lam
            phase = "full"

        total, ce, sp = train(model, train_loader, optimizer, current_lam, DEVICE)
        acc    = evaluate(model, test_loader, DEVICE)
        w_spar = calculate_sparsity(model)
        n_spar = neuron_sparsity(model)

        scheduler.step()

        print(f"  {epoch:>3}  {phase:>5}  {current_lam:>6.3f}  "
              f"{total:>7.4f}  {ce:>7.4f}  {sp:>7.4f}  "
              f"{acc:>6.1f}%  {w_spar:>7.1f}%  {n_spar:>7.1f}%")

    # End of run
    acc    = evaluate(model, test_loader, DEVICE)
    w_spar = calculate_sparsity(model)
    n_spar = neuron_sparsity(model)
    results.append((lam, acc, w_spar, n_spar))

    print(f"\n  Final → Acc: {acc:.2f}%  |  Weight-Sparsity: {w_spar:.2f}%"
          f"  |  Neuron-Sparsity: {n_spar:.2f}%")
    print_layer_stats(model)
    plot_gates(model, lam=lam)

# ── Summary ──────────────────────────────────────────────────────────────────
print_summary(results)