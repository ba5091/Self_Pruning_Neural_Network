"""
src/model.py

ROOT CAUSE OF ZERO SPARSITY — and why this version fixes it
─────────────────────────────────────────────────────────────
Weight-level gating (one gate per weight) has 1.57 M gate parameters in
layer 1 alone.  The L1 gradient per gate_score is:

    ∂L1/∂gate_score_i  =  λ × |w_i| × σ'(gs_i) / N_layer
                        ≈  λ × 0.015 × 0.106 / 1,572,864
                        ≈  λ × 1e-9

The CE gradient on the same gate_score is ≈ 1e-5.
Ratio ≈ λ × 1e-4  →  even λ=20 gives 0.2%.
Adam normalises by gradient history, so L1 is permanently drowned out.
Sparsity stays at the initialisation value (~2%) no matter what.

FIX: Neuron-level gating — one gate per OUTPUT NEURON.
    Layer 1: 512 gates  (not 1,572,864)
    Layer 2: 256 gates
    Layer 3: 128 gates
    Layer 4:  10 gates
    Total:   906 gates  vs 1,702,154 before

New gradient per gate_score:
    ∂L1/∂gate_score_i  ≈  λ × σ'(gs_i) / 906
                        ≈  λ × 1.17e-4

CE gradient ≈ 1.6e-4  →  Ratio ≈ λ × 0.7
At λ=1: L1 gradient is 70% the size of CE gradient → genuine competition.
Sparsity now meaningfully responds to λ.

Bonus: pruning whole neurons is more useful than pruning random weights —
it actually reduces inference FLOPs when rows with gate≈0 are removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronGatedLinear(nn.Module):
    """
    Linear layer with ONE learnable gate per OUTPUT neuron.

    Effective weight matrix = W * diag(gates)
    where gates = sigmoid(gate_scores) ∈ (0, 1)^out_features

    A neuron is "pruned" when its gate → 0, zeroing its entire row.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_features))

        # One gate_score per output neuron.
        # Init = 2.0  →  sigmoid(2) ≈ 0.88  (gates start fully open)
        # L1 pressure will push gate_scores negative (gates close toward 0).
        # CE loss fights to keep useful neurons open.
        self.gate_scores = nn.Parameter(
            torch.full((out_features,), 2.0)
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def gates(self) -> torch.Tensor:
        """Gate values in (0,1), shape (out_features,). Differentiable."""
        return torch.sigmoid(self.gate_scores)

    def effective_weights(self) -> torch.Tensor:
        """
        W_eff[i, j] = W[i, j] * gate[i]
        Broadcasting: gates (out,) → (out, 1) × W (out, in) = (out, in)
        """
        return self.weight * self.gates().unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equivalent to: gates applied after the linear, before bias.
        # Using effective_weights keeps gradient graph connected to gate_scores.
        out = F.linear(x, self.effective_weights())
        # Scale bias by gate too — a dead neuron contributes nothing.
        return out + self.bias * self.gates()

    # ------------------------------------------------------------------
    # Post-training utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def hard_prune(self, threshold: float = 0.5) -> int:
        """Zero rows whose gate < threshold. Call after training only."""
        mask = self.gates() < threshold        # (out,)
        self.weight[mask, :] = 0.0
        self.gate_scores[mask] = -20.0         # sigmoid(-20) ≈ 0
        return int(mask.sum().item())

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"gates={self.out_features}")


# ---------------------------------------------------------------------------

class PrunableNN(nn.Module):
    """
    4-layer MLP for CIFAR-10: 3072 → 512 → 256 → 128 → 10
    All linear layers use NeuronGatedLinear (906 total gate parameters).
    BatchNorm stabilises training while gates are being suppressed.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            NeuronGatedLinear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            NeuronGatedLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            NeuronGatedLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            NeuronGatedLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def gated_layers(self):
        for m in self.modules():
            if isinstance(m, NeuronGatedLinear):
                yield m

    def parameter_groups(self, weight_lr: float = 1e-3,
                         gate_lr: float = 5e-3) -> list:
        """
        Separate LR for gates vs weights.
        Gates need a higher LR so they can react to λ pressure
        without forcing all weights to update at gate speed.
        """
        weights, gates = [], []
        for name, param in self.named_parameters():
            (gates if 'gate_scores' in name else weights).append(param)
        return [
            {'params': weights, 'lr': weight_lr, 'name': 'weights'},
            {'params': gates,   'lr': gate_lr,   'name': 'gates'},
        ]

    @torch.no_grad()
    def hard_prune_all(self, threshold: float = 0.5) -> dict:
        return {
            f'layer_{i}': layer.hard_prune(threshold)
            for i, layer in enumerate(self.gated_layers())
        }