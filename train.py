"""
src/train.py

Why L1 is on GATES (not effective weights) in this version
────────────────────────────────────────────────────────────
With neuron-level gating, L1 on gates directly gives:

    gradient per gate_score  =  λ × σ'(gs) / N_gates
                              ≈  λ × 0.106 / 906
                              ≈  λ × 1.17e-4

CE gradient on same gate_score ≈ 1.6e-4.
At λ=1: ratio ≈ 0.73  →  L1 and CE genuinely compete.  Sparsity increases.

Contrast with weight-level L1 on |w×g|:
    gradient per gate_score  ≈  λ × 1e-9   (divided by 1.57M)
    CE gradient              ≈  1.6e-5
    Ratio ≈ λ × 1e-4  →  needs λ>10000 to matter.  Sparsity stays at 2%.

Additionally, L1 on gates is compensation-proof: the weight cannot prevent
a gate from being pushed toward 0, because the gate is a separate parameter.
"""

import torch
import torch.nn as nn

from src.model import NeuronGatedLinear


def sparsity_loss(model: nn.Module, lam: float) -> torch.Tensor:
    """
    L1 penalty on gate values: λ × mean(sigmoid(gate_scores)) across all
    gate parameters in the network.

    Calibration (906 total gate params, init gate≈0.88):
        penalty at init ≈ λ × 0.88
        CE at init      ≈ 2.3
        λ=0.5 → penalty ≈ 19% of CE  (mild pruning)
        λ=1.5 → penalty ≈ 57% of CE  (moderate pruning)
        λ=4.0 → penalty ≈ 150% of CE (aggressive, needs warmup)
    """
    device = next(model.parameters()).device
    all_gates = []

    for m in model.modules():
        if isinstance(m, NeuronGatedLinear):
            all_gates.append(m.gates())            # (out_features,)

    if not all_gates:
        return torch.tensor(0.0, device=device)

    # Concatenate all 906 gate values and take a single global mean.
    # This is valid because all gates are on the same (0,1) scale.
    return lam * torch.cat(all_gates).mean()


def train(model: nn.Module,
          loader,
          optimizer: torch.optim.Optimizer,
          lam: float,
          device: torch.device) -> tuple[float, float, float]:
    """
    One training epoch.
    Returns (total_loss, ce_loss, sparsity_loss) averaged over batches.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_sum = ce_sum = sp_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        ce_loss = criterion(model(images), labels)
        sp_loss = sparsity_loss(model, lam)
        loss    = ce_loss + sp_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_sum += loss.item()
        ce_sum    += ce_loss.item()
        sp_sum    += sp_loss.item()

    n = len(loader)
    return total_sum / n, ce_sum / n, sp_sum / n


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total