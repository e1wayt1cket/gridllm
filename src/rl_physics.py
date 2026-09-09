# rl_physics.py
"""Physics-guided interaction for the centralized critics (train-only).

Injects a fixed, offline electrical-distance interaction structure into a
centralized critic:
    h_i = f(o_i, a_i) + sum_{j != i} W[i, j] f(o_j, a_j)
with W a row-normalized exp(-distance / sigma) kernel over the storage agents'
buses. The weights depend only on network topology / series resistance (never
on line thermal limits), so W is invariant across the comparison scenarios and
can be precomputed once per bus set. Saved policies are actors only, so a
critic-only change never touches eval or checkpoints.
"""

import heapq
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn

from grid import build_base_network  # acyclic: grid imports no rl module


def _undirected_line_graph(net) -> dict:
    """Adjacency {bus: [(neighbor, resistance_ohm), ...]} over net.line.

    Uses series resistance r_ohm_per_km * length_km as the edge weight.
    """
    adj = defaultdict(list)
    for _, row in net.line.iterrows():
        f = int(row["from_bus"])
        t = int(row["to_bus"])
        r = float(row["r_ohm_per_km"]) * float(row["length_km"])
        adj[f].append((t, r))
        adj[t].append((f, r))
    return adj


def pairwise_resistance_distance(net, buses: List[int]) -> np.ndarray:
    """Pairwise shortest-path series resistance between buses (Dijkstra).

    The 33-bus feeder is a radial tree so the path is unique; Dijkstra is
    used regardless for safety. Returns an (n_buses, n_buses) matrix ordered
    like the input list. Unreachable pairs are inf.
    """
    adj = _undirected_line_graph(net)
    buses = list(buses)
    D = np.full((len(buses), len(buses)), np.inf)
    for i, src in enumerate(buses):
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            for v, w in adj.get(u, ()):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        for j, dst in enumerate(buses):
            D[i, j] = dist.get(dst, float("inf"))
    return D


def storage_interaction_weights(config, buses: List[int],
                                sigma: float = None) -> np.ndarray:
    """Row-normalized exp(-distance/sigma) kernel between storage buses.

    Zero diagonal. When sigma is None it defaults to the maximum pairwise
    distance so weights span a comparable scale across networks.
    """
    net = build_base_network(config)
    D = pairwise_resistance_distance(net, buses)
    if sigma is None:
        finite = D[np.isfinite(D)]
        sigma = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    W = np.exp(-np.asarray(D, dtype=float) / sigma)
    np.fill_diagonal(W, 0.0)
    row_sum = W.sum(axis=1, keepdims=True)
    W = np.where(row_sum > 0, W / np.maximum(row_sum, 1e-9), W)
    return W


class PhysicsCentralizedCritic(nn.Module):
    """Twin critic over a physics-mixed per-agent context.

    Encoder f maps each agent's (o_i, a_i) to a context embedding; agent i's
    context is its own embedding plus the W-weighted sum of all other agents'
    embeddings, then twin Q heads output out_dim values (1 for MATD3, N
    quantiles for qmatd3). W and agent_idx fix the network geometry at build
    time; target copies share the same geometry (load_state_dict compatible).
    """

    def __init__(self, n_agents: int, obs_dim: int, act_dim: int,
                 W: np.ndarray, agent_idx: int, out_dim: int = 1,
                 h: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.agent_idx = agent_idx
        self.out_dim = out_dim
        self.register_buffer("W",
                             torch.as_tensor(np.asarray(W, dtype=np.float32)))
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU())
        half = max(h // 2, 1)
        def head():
            return nn.Sequential(
                nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(h, half), nn.ReLU(),
                nn.Linear(half, out_dim))
        self.q1 = head()
        self.q2 = head()

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """obs (B, obs*n), actions (B, act*n) -> (q1, q2) each (B, out_dim)."""
        B = obs.shape[0]
        o = obs.view(B, self.n_agents, -1)
        a = actions.view(B, self.n_agents, -1)
        xa = torch.cat([o, a], dim=-1)                # (B, n, obs+act)
        emb = self.encoder(xa)                        # (B, n, h), shared encoder
        w = self.W[self.agent_idx].view(1, self.n_agents, 1)
        ctx = emb[:, self.agent_idx] + (w * emb).sum(dim=1)   # (B, h)
        return self.q1(ctx), self.q2(ctx)
