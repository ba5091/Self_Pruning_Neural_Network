"""
src/utils.py

Sparsity definition for neuron-level gating
─────────────────────────────────────────────
A NEURON is pruned when gate = sigmoid(gate_score) < GATE_THRESHOLD (0.5).
All weights in that neuron's row are effectively zeroed.

Sparsity (%) = (weights in pruned neuron rows) / (total weights) × 100

Why gate < 0.5 (not |w×g| < 1e-3):
  - With neuron-level gates the natural binary criterion is gate ≷ 0.5.
  - |w×g| < 1e-3 required gate × weight < 1e-3. For a gate of 0.3 and
    typical weight 0.04, |w×g| = 0.012 >> 1e-3 — still reported as active.
  - gate < 0.5 directly captures "this neuron's output is suppressed below
    half its potential", which is the meaningful pruning criterion.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.model import NeuronGatedLinear

GATE_THRESHOLD = 0.5    # neuron pruned if sigmoid(gate_score) < this


# ---------------------------------------------------------------------------
# Sparsity measurement
# ---------------------------------------------------------------------------

def calculate_sparsity(model: nn.Module,
                       threshold: float = GATE_THRESHOLD) -> float:
    """
    Global sparsity: percentage of weight parameters belonging to
    neurons whose gate < threshold.
    """
    total_weights  = 0
    pruned_weights = 0

    for m in model.modules():
        if isinstance(m, NeuronGatedLinear):
            g = m.gates().detach()
            pruned_rows    = (g < threshold).sum().item()
            total_weights  += m.weight.numel()
            pruned_weights += pruned_rows * m.in_features   # full row is zeroed

    return 100.0 * pruned_weights / total_weights if total_weights > 0 else 0.0


def neuron_sparsity(model: nn.Module,
                    threshold: float = GATE_THRESHOLD) -> float:
    """Percentage of neurons (not weights) that are pruned."""
    total = pruned = 0
    for m in model.modules():
        if isinstance(m, NeuronGatedLinear):
            g = m.gates().detach()
            total  += g.numel()
            pruned += (g < threshold).sum().item()
    return 100.0 * pruned / total if total > 0 else 0.0


def layer_stats(model: nn.Module,
                threshold: float = GATE_THRESHOLD) -> list[dict]:
    """Per-layer breakdown for diagnostics."""
    stats = []
    for i, m in enumerate(model.modules()):
        if isinstance(m, NeuronGatedLinear):
            g = m.gates().detach()
            pruned = (g < threshold).sum().item()
            stats.append({
                'layer'          : i,
                'neurons'        : m.out_features,
                'pruned_neurons' : pruned,
                'neuron_sparsity': 100.0 * pruned / m.out_features,
                'mean_gate'      : g.mean().item(),
                'min_gate'       : g.min().item(),
            })
    return stats


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_gates(model: nn.Module, lam: float = None, save: bool = True) -> None:
    """
    3-panel diagnostic plot:
      Left   — gate value distribution across all 906 neurons
               (should shift left from 0.88 toward 0 as training progresses)
      Centre — per-layer mean gate value (bar chart)
      Right  — per-layer % of neurons pruned (bar chart)
    """
    stats = layer_stats(model)
    all_gates = []
    for m in model.modules():
        if isinstance(m, NeuronGatedLinear):
            all_gates.append(m.gates().detach().cpu().numpy())

    if not all_gates:
        print("No NeuronGatedLinear layers found.")
        return

    all_gates    = np.concatenate(all_gates)
    pct_neurons  = 100.0 * (all_gates < GATE_THRESHOLD).mean()
    title_suffix = f"  λ={lam}" if lam is not None else ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Gate distribution
    axes[0].hist(all_gates, bins=50, color='steelblue', edgecolor='none')
    axes[0].axvline(GATE_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                    label=f'threshold={GATE_THRESHOLD}')
    axes[0].set_xlabel('Gate Value  σ(gate_score)')
    axes[0].set_ylabel('Neuron count')
    axes[0].set_xlim(0, 1)
    axes[0].set_title(
        f'Gate Distribution{title_suffix}\n'
        f'{pct_neurons:.1f}% of neurons pruned'
    )
    axes[0].legend(fontsize=8)

    # 2. Per-layer mean gate
    labels     = [f"L{s['layer']}\n({s['neurons']} neurons)" for s in stats]
    mean_gates = [s['mean_gate'] for s in stats]
    axes[1].bar(labels, mean_gates, color='darkorange', edgecolor='none')
    axes[1].axhline(GATE_THRESHOLD, color='red', linestyle='--', linewidth=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Mean gate value')
    axes[1].set_title(f'Mean Gate per Layer{title_suffix}')
    for x, v in enumerate(mean_gates):
        axes[1].text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    # 3. Per-layer neuron sparsity
    sparsities = [s['neuron_sparsity'] for s in stats]
    bars = axes[2].bar(labels, sparsities, color='mediumseagreen', edgecolor='none')
    for bar, v in zip(bars, sparsities):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1, f'{v:.1f}%',
                     ha='center', va='bottom', fontsize=9)
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel('Neurons pruned (%)')
    axes[2].set_title(f'Neuron Sparsity per Layer{title_suffix}')

    plt.tight_layout()
    if save:
        fname = f"gate_dist_lam{lam}.png" if lam else "gate_dist.png"
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        print(f"  Plot saved → {fname}")
    plt.show()


def print_layer_stats(model: nn.Module) -> None:
    print(f"\n  {'L':>3}  {'Neurons':>8}  {'Pruned':>7}  {'N-Spar':>8}  {'MeanGate':>9}  {'MinGate':>8}")
    print(f"  {'-'*50}")
    for s in layer_stats(model):
        print(f"  {s['layer']:>3}  {s['neurons']:>8}  {s['pruned_neurons']:>7}"
              f"  {s['neuron_sparsity']:>7.1f}%  {s['mean_gate']:>9.4f}  {s['min_gate']:>8.4f}")
    print()


def print_summary(results: list[tuple]) -> None:
    print("\n" + "=" * 60)
    print(f"  {'λ':>5}  {'Accuracy':>10}  {'W-Sparsity':>12}  {'N-Sparsity':>12}")
    print(f"  {'-'*55}")
    for item in results:
        lam, acc, w_spar, n_spar = item
        print(f"  {lam:>5}  {acc:>9.2f}%  {w_spar:>11.2f}%  {n_spar:>11.2f}%")
    print("=" * 60 + "\n")