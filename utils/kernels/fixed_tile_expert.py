"""Optional FP16 expert kernel with fixed per-output reduction tiles.

The row count controls only the grid and masked loads. Each real output uses
identical K-loop and tile dimensions across regrouping/placement changes.
This is a distinct numerical version; it need not match native cuBLAS outputs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up(X, G, U, H, M, K: tl.constexpr, N: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    ag = tl.zeros((BM, BN), tl.float32)
    au = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + inner
        x = tl.load(X + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), other=0)
        gate = tl.load(G + cols[None, :] * K + ks[:, None],
                       (cols[None, :] < N) & (ks[:, None] < K), other=0)
        up = tl.load(U + cols[None, :] * K + ks[:, None],
                     (cols[None, :] < N) & (ks[:, None] < K), other=0)
        ag = tl.dot(x, gate, ag, allow_tf32=False)
        au = tl.dot(x, up, au, allow_tf32=False)
    # Preserve the FP16 intermediate boundaries of linear -> SiLU -> multiply.
    g = ag.to(tl.float16).to(tl.float32)
    u = au.to(tl.float16).to(tl.float32)
    activation = (g / (1.0 + tl.exp(-g))).to(tl.float16).to(tl.float32)
    value = (activation * u).to(tl.float16)
    tl.store(H + rows[:, None] * N + cols[None, :], value,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _down(X, W, Y, M, K: tl.constexpr, N: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + inner
        x = tl.load(X + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), other=0)
        w = tl.load(W + cols[None, :] * K + ks[:, None],
                    (cols[None, :] < N) & (ks[:, None] < K), other=0)
        acc = tl.dot(x, w, acc, allow_tf32=False)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc.to(tl.float16),
             (rows[:, None] < M) & (cols[None, :] < N))


def expert(x, gate, up, down, *, row_tile=64):
    tensors = (x, gate, up, down)
    assert all(t.is_cuda and t.dtype == torch.float16 and t.is_contiguous()
               and t.device == x.device and t.ndim == 2 for t in tensors)
    assert row_tile in (64, 128)
    warps = 4 if row_tile == 64 else 8
    m, k = x.shape
    n = gate.shape[0]
    assert gate.shape == up.shape == (n, k) and down.shape == (k, n)
    result = torch.empty((m, k), device=x.device, dtype=x.dtype)
    if m == 0:
        return result
    hidden = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = lambda width: (triton.cdiv(m, row_tile), triton.cdiv(width, 64))
    _gate_up[grid(n)](x, gate, up, hidden, m, k, n, row_tile, 64, 32,
                      num_warps=warps, num_stages=2, enable_fp_fusion=False)
    _down[grid(k)](hidden, down, result, m, n, k, row_tile, 64, 32,
                   num_warps=warps, num_stages=2, enable_fp_fusion=False)
    return result
