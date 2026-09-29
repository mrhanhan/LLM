"""神经元点云：把权重矩阵的行/列用 PCA 投到 3D。"""
from __future__ import annotations

import torch


def neuron_cloud(weight: torch.Tensor, max_nodes: int = 192, top_edges: int = 2500,
                 seed: int = 0) -> dict:
    """把 [out, in] 权重矩阵变成可画点云的节点 + 连线。

    - 输出神经元：权重矩阵的行向量，PCA 到 3D
    - 输入神经元：权重矩阵的列向量（转置），PCA 到 3D
    - 连线：|w_ij| 最大的若干条，粗细 ∝ |w|，颜色按正负
    """
    w = weight.detach().float()
    out_dim, in_dim = w.shape
    g = torch.Generator().manual_seed(seed)

    def pick(n):
        if n <= max_nodes:
            return torch.arange(n)
        return torch.randperm(n, generator=g)[:max_nodes].sort().values

    out_idx = pick(out_dim)
    in_idx = pick(in_dim)
    sub = w[out_idx][:, in_idx]  # [On, In]

    out_xyz = _pca3(sub)          # 行向量
    in_xyz = _pca3(sub.t())       # 列向量

    # 连线：取 |w| 最大的 top_edges 条
    flat = sub.abs().reshape(-1)
    k = min(top_edges, flat.numel())
    vals, idxs = torch.topk(flat, k)
    oi = (idxs // sub.shape[1]).tolist()
    ii = (idxs % sub.shape[1]).tolist()
    wv = sub.reshape(-1)[idxs].tolist()

    return {
        "out_coords": out_xyz.tolist(),
        "in_coords": in_xyz.tolist(),
        "out_idx": out_idx.tolist(),
        "in_idx": in_idx.tolist(),
        "out_norm": sub.norm(dim=1).tolist(),
        "in_norm": sub.norm(dim=0).tolist(),
        "edges": [{"o": o, "i": i, "w": v} for o, i, v in zip(oi, ii, wv)],
    }


def _pca3(mat: torch.Tensor) -> torch.Tensor:
    """把 [N, D] 的向量用 SVD 投影到 3D，并归一化到 [-1, 1]。"""
    x = mat - mat.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return torch.zeros(x.shape[0], 3)
    q = min(3, x.shape[0], x.shape[1])
    try:
        u, s, _ = torch.svd_lowrank(x, q=q)
        coords = u[:, :3] * s[:3]
    except Exception:
        coords = torch.zeros(x.shape[0], 3)
    if coords.shape[1] < 3:
        pad = torch.zeros(coords.shape[0], 3 - coords.shape[1])
        coords = torch.cat([coords, pad], dim=1)
    scale = coords.abs().max()
    if scale > 0:
        coords = coords / scale
    return coords
