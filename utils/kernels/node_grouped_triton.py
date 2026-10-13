from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on machines without Triton.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    return bool(_TRITON_AVAILABLE)


def can_use_node_grouped_triton(idx: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        bool(_TRITON_AVAILABLE)
        and idx.is_cuda
        and w.is_cuda
        and idx.dim() == 2
        and int(idx.shape[1]) <= 8
    )


def can_use_owner_payload_triton(hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> bool:
    return (
        bool(_TRITON_AVAILABLE)
        and hidden.is_cuda
        and ids.is_cuda
        and weights.is_cuda
        and hidden.dim() == 2
        and ids.dim() == 2
        and weights.dim() == 2
        and int(ids.shape[1]) <= 8
        and int(weights.shape[1]) == int(ids.shape[1])
        and int(hidden.shape[0]) == int(ids.shape[0]) == int(weights.shape[0])
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _pack_node_packets_kernel(
        idx_ptr,
        w_ptr,
        expert_to_rank_ptr,
        rank_to_node_ptr,
        node_reps_ptr,
        out_dest_ptr,
        out_token_ptr,
        out_i_ptr,
        out_w_ptr,
        T: tl.constexpr,
        K: tl.constexpr,
        NUM_NODES: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        t = tl.program_id(0)
        node = tl.program_id(1)
        offs = tl.arange(0, BLOCK_K)
        valid_k = offs < K

        e = tl.load(idx_ptr + t * K + offs, mask=valid_k, other=0).to(tl.int64)
        rank = tl.load(expert_to_rank_ptr + e, mask=valid_k, other=0).to(tl.int64)
        e_node = tl.load(rank_to_node_ptr + rank, mask=valid_k, other=-1).to(tl.int64)
        match = valid_k & (e_node == node)
        slot = tl.cumsum(match.to(tl.int32), 0) - 1
        present = tl.sum(match.to(tl.int32), axis=0) > 0

        row = t * NUM_NODES + node
        tl.store(out_i_ptr + row * K + offs, -1, mask=valid_k)
        tl.store(out_w_ptr + row * K + offs, 0.0, mask=valid_k)
        tl.store(out_i_ptr + row * K + slot, e, mask=match)
        ww = tl.load(w_ptr + t * K + offs, mask=valid_k, other=0.0)
        tl.store(out_w_ptr + row * K + slot, ww, mask=match)
        rep = tl.load(node_reps_ptr + node)
        tl.store(out_dest_ptr + row, tl.where(present, rep, -1))
        tl.store(out_token_ptr + row, t)

    @triton.jit
    def _pack_rank_packets_kernel(
        recv_i_ptr,
        recv_w_ptr,
        expert_to_rank_ptr,
        out_dest_ptr,
        out_packet_ptr,
        out_i_ptr,
        out_w_ptr,
        P: tl.constexpr,
        K: tl.constexpr,
        COMM_WORLD: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        p = tl.program_id(0)
        rank_slot = tl.program_id(1)
        offs = tl.arange(0, BLOCK_K)
        valid_k = offs < K

        e_raw = tl.load(recv_i_ptr + p * K + offs, mask=valid_k, other=-1).to(tl.int64)
        valid_e = valid_k & (e_raw >= 0)
        e_safe = tl.where(valid_e, e_raw, 0)
        dest_rank = tl.load(expert_to_rank_ptr + e_safe, mask=valid_k, other=-1).to(tl.int64)
        match = valid_e & (dest_rank == rank_slot)
        slot = tl.cumsum(match.to(tl.int32), 0) - 1
        present = tl.sum(match.to(tl.int32), axis=0) > 0

        row = p * COMM_WORLD + rank_slot
        tl.store(out_i_ptr + row * K + offs, -1, mask=valid_k)
        tl.store(out_w_ptr + row * K + offs, 0.0, mask=valid_k)
        tl.store(out_i_ptr + row * K + slot, e_raw, mask=match)
        ww = tl.load(recv_w_ptr + p * K + offs, mask=valid_k, other=0.0)
        tl.store(out_w_ptr + row * K + slot, ww, mask=match)
        tl.store(out_dest_ptr + row, tl.where(present, rank_slot, -1))
        tl.store(out_packet_ptr + row, p)

    @triton.jit
    def _pack_owner_payload_kernel(
        hidden_ptr,
        ids_ptr,
        weights_ptr,
        payload_ptr,
        N: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offs = block * BLOCK_W + tl.arange(0, BLOCK_W)
        valid = (row < N) & (offs < WIDTH)

        hidden_mask = valid & (offs < H)
        hidden_vals = tl.load(hidden_ptr + row * H + offs, mask=hidden_mask, other=0.0)
        tl.store(payload_ptr + row * WIDTH + offs, hidden_vals, mask=hidden_mask)

        id_offs = offs - H
        id_mask = valid & (offs >= H) & (offs < H + K)
        id_vals = tl.load(ids_ptr + row * K + id_offs, mask=id_mask, other=-1).to(tl.float32)
        tl.store(payload_ptr + row * WIDTH + offs, id_vals, mask=id_mask)

        weight_offs = offs - H - K
        weight_mask = valid & (offs >= H + K) & (offs < H + 2 * K)
        weight_vals = tl.load(weights_ptr + row * K + weight_offs, mask=weight_mask, other=0.0)
        tl.store(payload_ptr + row * WIDTH + offs, weight_vals, mask=weight_mask)

    @triton.jit
    def _unpack_owner_payload_kernel(
        payload_ptr,
        hidden_ptr,
        ids_ptr,
        weights_ptr,
        N: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offs = block * BLOCK_W + tl.arange(0, BLOCK_W)
        valid = (row < N) & (offs < WIDTH)
        vals = tl.load(payload_ptr + row * WIDTH + offs, mask=valid, other=0.0)

        hidden_mask = valid & (offs < H)
        tl.store(hidden_ptr + row * H + offs, vals, mask=hidden_mask)

        id_offs = offs - H
        id_mask = valid & (offs >= H) & (offs < H + K)
        id_vals = vals.to(tl.float32).to(tl.int64)
        tl.store(ids_ptr + row * K + id_offs, id_vals, mask=id_mask)

        weight_offs = offs - H - K
        weight_mask = valid & (offs >= H + K) & (offs < H + 2 * K)
        tl.store(weights_ptr + row * K + weight_offs, vals, mask=weight_mask)


def _block_k(k: int) -> int:
    if k <= 1:
        return 1
    if k <= 2:
        return 2
    if k <= 4:
        return 4
    return 8


def _block_w(width: int) -> int:
    if width <= 256:
        return 256
    if width <= 512:
        return 512
    return 1024


def pack_node_packets_triton(
    idx: torch.Tensor,
    w: torch.Tensor,
    expert_to_rank: torch.Tensor,
    rank_to_node: torch.Tensor,
    node_reps: torch.Tensor,
    *,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack one row per ``(token, destination node)`` using Triton.

    The kernel writes a dense ``T * num_nodes`` workspace and compacts present
    rows with a device boolean mask. This avoids CPU list/dict grouping while
    keeping the external runtime semantics identical to the PyTorch path.
    """
    if not can_use_node_grouped_triton(idx, w):
        raise RuntimeError("Triton node-grouped pack is not available for these tensors")
    if int(num_nodes) <= 0:
        raise RuntimeError(f"invalid num_nodes={num_nodes}")
    T = int(idx.shape[0])
    K = int(idx.shape[1])
    total = int(T * int(num_nodes))
    device = idx.device
    out_dest = torch.empty((total,), device=device, dtype=torch.long)
    out_token = torch.empty((total,), device=device, dtype=torch.long)
    out_i = torch.empty((total, K), device=device, dtype=torch.long)
    out_w = torch.empty((total, K), device=device, dtype=w.dtype)

    _pack_node_packets_kernel[(T, int(num_nodes))](
        idx,
        w,
        expert_to_rank,
        rank_to_node,
        node_reps,
        out_dest,
        out_token,
        out_i,
        out_w,
        T,
        K,
        int(num_nodes),
        BLOCK_K=_block_k(K),
    )
    mask = out_dest.ge(0)
    return out_dest[mask].contiguous(), out_token[mask].contiguous(), out_i[mask].contiguous(), out_w[mask].contiguous()


def pack_rank_packets_triton(
    recv_node_i: torch.Tensor,
    recv_node_w: torch.Tensor,
    expert_to_rank: torch.Tensor,
    *,
    comm_world: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split node packets into one row per ``(node packet, destination rank)``."""
    if not can_use_node_grouped_triton(recv_node_i, recv_node_w):
        raise RuntimeError("Triton rank-pack is not available for these tensors")
    P = int(recv_node_i.shape[0])
    K = int(recv_node_i.shape[1])
    total = int(P * int(comm_world))
    device = recv_node_i.device
    out_dest = torch.empty((total,), device=device, dtype=torch.long)
    out_packet = torch.empty((total,), device=device, dtype=torch.long)
    out_i = torch.empty((total, K), device=device, dtype=torch.long)
    out_w = torch.empty((total, K), device=device, dtype=recv_node_w.dtype)

    _pack_rank_packets_kernel[(P, int(comm_world))](
        recv_node_i,
        recv_node_w,
        expert_to_rank,
        out_dest,
        out_packet,
        out_i,
        out_w,
        P,
        K,
        int(comm_world),
        BLOCK_K=_block_k(K),
    )
    mask = out_dest.ge(0)
    return out_dest[mask].contiguous(), out_packet[mask].contiguous(), out_i[mask].contiguous(), out_w[mask].contiguous()


def pack_owner_payload_triton(
    hidden: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    payload_dtype: torch.dtype,
) -> torch.Tensor:
    """Fuse hidden/id/weight rows into one owner-preserving dispatch payload."""
    if not can_use_owner_payload_triton(hidden, ids, weights):
        raise RuntimeError("Triton owner-payload pack is not available for these tensors")
    N = int(hidden.shape[0])
    H = int(hidden.shape[1])
    K = int(ids.shape[1])
    width = int(H + 2 * K)
    payload = torch.empty((N, width), device=hidden.device, dtype=payload_dtype)
    block_w = _block_w(width)
    grid = (N, triton.cdiv(width, block_w))
    _pack_owner_payload_kernel[grid](
        hidden,
        ids,
        weights,
        payload,
        N,
        H,
        K,
        width,
        BLOCK_W=block_w,
    )
    return payload


def unpack_owner_payload_triton(
    payload: torch.Tensor,
    *,
    hidden_dim: int,
    top_k: int,
    activation_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split an owner-preserving dispatch payload back into hidden/id/weight rows."""
    if not bool(_TRITON_AVAILABLE) or not payload.is_cuda or payload.dim() != 2:
        raise RuntimeError("Triton owner-payload unpack is not available for this tensor")
    N = int(payload.shape[0])
    H = int(hidden_dim)
    K = int(top_k)
    width = int(H + 2 * K)
    if int(payload.shape[1]) != width:
        raise RuntimeError(f"invalid owner payload width={int(payload.shape[1])}, expected={width}")
    hidden = torch.empty((N, H), device=payload.device, dtype=activation_dtype)
    ids = torch.empty((N, K), device=payload.device, dtype=torch.long)
    weights = torch.empty((N, K), device=payload.device, dtype=activation_dtype)
    block_w = _block_w(width)
    grid = (N, triton.cdiv(width, block_w))
    _unpack_owner_payload_kernel[grid](
        payload,
        hidden,
        ids,
        weights,
        N,
        H,
        K,
        width,
        BLOCK_W=block_w,
    )
    return hidden, ids, weights
