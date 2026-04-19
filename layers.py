import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Initialize weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores
        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )

    def forward(self, x):
        # 🔥 Temperature scaling (controls sharpness of pruning)
        temperature = 5

        # Convert scores → gates (0 to 1)
        gates = torch.sigmoid(self.gate_scores * temperature)

        # Apply pruning
        pruned_weights = self.weight * gates

        # Linear transformation
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        # Same transformation for evaluation
        temperature = 5
        return torch.sigmoid(self.gate_scores * temperature)