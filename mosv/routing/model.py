import torch
import torch.nn as nn
import torch.nn.functional as F


class MoSVRouter(nn.Module):
    """3-layer MLP router with residual skip: prompt hidden state → sparse weights over K steering vectors."""

    def __init__(
        self,
        d_model: int,
        K: int,
        top_k: int = 2,
        dropout: float = 0.3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.K = K
        self.top_k = top_k
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(d_model, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, K, bias=True)
        self.skip = nn.Linear(d_model, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.fc1, self.fc2, self.fc3, self.skip]:
            nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1) + self.skip(x))
        return self.fc3(h2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model]
        Returns:
            weights: [batch, K]  sparse (top_k non-zero entries per row)
        """
        x = self.dropout(x)
        logits = self._mlp(x)

        top_k = min(self.top_k, self.K)
        top_values, top_indices = logits.topk(top_k, dim=-1)

        sparse = torch.zeros_like(logits)
        sparse.scatter_(-1, top_indices, F.softmax(top_values, dim=-1))
        return sparse

    def route_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (used for training with cross-entropy)."""
        return self._mlp(self.dropout(x))
