from __future__ import annotations

import torch
from torch import nn


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _norm_adj(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        self_loop = torch.arange(num_nodes, device=edge_index.device)
        row = torch.cat([row, self_loop])
        col = torch.cat([col, self_loop])
        val = torch.ones(row.numel(), device=edge_index.device)
        A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (num_nodes, num_nodes)).coalesce()
        deg = torch.sparse.sum(A, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        v = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(torch.stack([row, col]), v, (num_nodes, num_nodes)).coalesce()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        A = self._norm_adj(x.size(0), edge_index)
        h = torch.sparse.mm(A, x)
        h = torch.relu(self.lin1(h))
        h = self.dropout(h)
        h = torch.sparse.mm(A, h)
        return self.lin2(h)
