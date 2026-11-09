"""BLOC C — Extracteur GNN pour SB3 (BaseFeaturesExtractor).

Pipeline (cf. plan §3) :
  1. L'observation à plat (167 dims) est redécoupée en graphe :
     features nœuds (33 x 3), features arêtes (32 x 2).
  2. 2 couches de convolution de graphe (GATConv par défaut, respecte les
     features d'arêtes ; option GCNConv).
  3. Compression : Global Mean Pooling + Global Max Pooling -> dense.
  4. Concatène le résumé spatial avec le prix du marché + SoC -> embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool

from src.envs.grid_env import (
    EDGE_FEATS,
    EDGE_SECTION,
    N_LINES,
    N_NODES,
    N_SITES,
    NODE_FEATS,
    NODE_SECTION,
)


class GNNFeatureExtractor(BaseFeaturesExtractor):
    """Obs (B, 167) -> graphe -> 2 convolutions -> pooling -> (B, `features_dim`)."""

    def __init__(
        self,
        observation_space: spaces.Box,
        edge_index: list[tuple[int, int]],
        conv_type: str = "gat",
        hidden_dim: int = 64,
        features_dim: int = 32,
        n_heads: int = 2,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        assert conv_type in ("gat", "gcn"), "conv_type: 'gat' | 'gcn'"

        self.conv_type = conv_type
        in_node = NODE_FEATS if conv_type == "gat" else NODE_FEATS + EDGE_FEATS

        if conv_type == "gat":
            assert hidden_dim % n_heads == 0, "hidden_dim divisible par n_heads"
            head_dim = hidden_dim // n_heads
            self.conv1 = GATConv(in_node, head_dim, heads=n_heads, edge_dim=EDGE_FEATS)
            self.conv2 = GATConv(hidden_dim, head_dim, heads=n_heads, edge_dim=EDGE_FEATS)
        else:
            self.conv1 = GCNConv(in_node, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.post_pool = nn.Linear(2 * hidden_dim, hidden_dim)  # mean ++ max pool
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1 + N_SITES, hidden_dim),  # + price + SoC
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
        )

        # Topologie constante du feeder radial -> buffer GPU/CPU
        ei = torch.tensor(list(edge_index), dtype=torch.long).t().contiguous()  # (2, E)
        self.register_buffer("edge_index_base", ei)
        self._edge_cache: dict[int, torch.Tensor] = {}

    def _tiled_edge_index(self, batch: int, device: torch.device) -> torch.Tensor:
        """edge_index déployé sur les `batch` graphes disjoint du mini-batch."""
        cached = self._edge_cache.get((batch, str(device)))
        if cached is None:
            offset = torch.arange(batch, device=device, dtype=torch.long) * N_NODES
            cached = torch.cat(
                [self.edge_index_base + off for off in offset], dim=1
            ).to(device)
            self._edge_cache[(batch, str(device))] = cached
        return cached

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        device = obs.device

        x = obs[:, :NODE_SECTION].reshape(batch, N_NODES, NODE_FEATS)
        e = obs[:, NODE_SECTION:EDGE_SECTION].reshape(batch, N_LINES, EDGE_FEATS)
        price = obs[:, EDGE_SECTION:EDGE_SECTION + 1]
        soc = obs[:, EDGE_SECTION + 1:]

        x_flat = x.reshape(batch * N_NODES, -1)
        batch_vec = torch.arange(batch, device=device).repeat_interleave(N_NODES)

        if self.conv_type == "gat":
            edge_attr = e.reshape(batch * N_LINES, -1)
            ei = self._tiled_edge_index(batch, device)
            x_flat = F.relu(self.conv1(x_flat, ei, edge_attr))
            x_flat = F.relu(self.conv2(x_flat, ei, edge_attr))
        else:
            # GCNConv ignore les arêtes : on injecte leurs features dans le nœud
            # cible (bus aval), cf. convention "la ligne alimente le bus suivant".
            ei = self._tiled_edge_index(batch, device)
            node_extra = torch.zeros(
                batch, N_NODES, EDGE_FEATS, device=device, dtype=obs.dtype
            )
            to_buses = self.edge_index_base[1]
            node_extra[:, to_buses, :] = e.to(device)
            x_flat = torch.cat([x_flat, node_extra.reshape(batch * N_NODES, -1)], dim=1)
            x_flat = F.relu(self.conv1(x_flat, ei))
            x_flat = F.relu(self.conv2(x_flat, ei))

        pooled = torch.cat(
            [global_mean_pool(x_flat, batch_vec), global_max_pool(x_flat, batch_vec)],
            dim=1,
        )
        h = F.relu(self.post_pool(pooled))
        return self.head(
            torch.cat([h, price.to(device), soc.to(device)], dim=1)
        )
