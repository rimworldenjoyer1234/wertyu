from __future__ import annotations

import torch
from torch import nn


class SimpleGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim, bias=False)
        self.att = nn.Parameter(torch.empty(2 * hidden_dim))
        self.out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.uniform_(self.att, -0.1, 0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        src, dst = edge_index
        hs, hd = h[src], h[dst]
        e = torch.leaky_relu((torch.cat([hs, hd], dim=1) * self.att).sum(dim=1), negative_slope=0.2)
        exp_e = torch.exp(e - e.max())
        denom = torch.zeros(x.size(0), device=x.device)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-9)
        msg = hs * alpha.unsqueeze(1)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        z = torch.relu(agg)
        z = self.dropout(z)
        return self.out(z)
