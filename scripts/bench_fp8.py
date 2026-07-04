"""H200 FP8 vs BF16 micro-benchmark.

Tests:
  1. Pure GEMM throughput at attention-like shapes (M=K=N=4096/8192)
  2. FP8 vs FP32 numerical error on random + synthetic matrices

Output: speedup ratio + error stats.  Tells us the *ceiling* of FP8 wins.
"""
from __future__ import annotations

import torch
import time

device = torch.device("cuda")

def bench_gemm(M, K, N, dtype, n_iter=200, warmup=20):
    """Benchmark M×K @ K×N at given dtype.  Returns TFLOPS."""
    if dtype == torch.float8_e4m3fn:
        a = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(dtype)
        b_T = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(dtype)
        b = b_T.t()
        scale_a = torch.tensor([1.0], device=device)
        scale_b = torch.tensor([1.0], device=device)
        # Warmup
        for _ in range(warmup):
            out, _ = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b,
                                       out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            out, _ = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b,
                                       out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n_iter
    else:
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)
        for _ in range(warmup):
            out = a @ b
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            out = a @ b
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n_iter

    tflops = (2 * M * K * N) / dt / 1e12
    return dt * 1e6, tflops  # microseconds, TFLOPS


def quality_test(M, K, N, distribution="normal"):
    """FP8 vs FP32 error on synthetic data."""
    if distribution == "normal":
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
    elif distribution == "softmax":
        # Attention-like: softmax output is positive and concentrated
        a = torch.softmax(torch.randn(M, K, device=device), dim=-1).float()
        b = torch.randn(K, N, device=device, dtype=torch.float32) * 0.1
    elif distribution == "uniform":
        a = (torch.rand(M, K, device=device) - 0.5) * 2
        b = (torch.rand(K, N, device=device) - 0.5) * 2

    ref = a @ b   # FP32 reference

    # FP8 path with automatic scaling
    a_amax = a.abs().max().item()
    b_amax = b.abs().max().item()
    # E4M3 max representable ~ 448
    s_a = 448.0 / max(a_amax, 1e-9)
    s_b = 448.0 / max(b_amax, 1e-9)
    a_fp8 = (a * s_a).to(torch.float8_e4m3fn)
    b_fp8 = (b * s_b).to(torch.float8_e4m3fn)
    scale_a = torch.tensor([1.0 / s_a], device=device)
    scale_b = torch.tensor([1.0 / s_b], device=device)
    out_fp8, _ = torch._scaled_mm(
        a_fp8, b_fp8.t().contiguous().t(),
        scale_a=scale_a, scale_b=scale_b,
        out_dtype=torch.float32,
    )
    abs_err = (out_fp8 - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-6)
    return {
        "mean_abs": abs_err.mean().item(),
        "max_abs": abs_err.max().item(),
        "mean_rel": rel_err.mean().item(),
        "p99_rel": rel_err.flatten().quantile(0.99).item(),
    }


def main():
    print("=" * 72)
    print("H200 FP8 vs BF16 GEMM SPEED (representative SAM attention shapes)")
    print("=" * 72)
    print(f"{'shape':<24} {'BF16 µs':>10} {'BF16 TFL':>10} {'FP8 µs':>10} {'FP8 TFL':>10} {'speedup':>10}")
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 4096, 4096),
        # SAM 3.1 attention rough sizes
        (1280, 4096, 4096),  # batch×seq × heads × dim
        (4096, 128, 4096),   # attention QK
        (4096, 4096, 128),   # attention AV
    ]
    for M, K, N in shapes:
        bf_us, bf_tfl = bench_gemm(M, K, N, torch.bfloat16, n_iter=50)
        fp_us, fp_tfl = bench_gemm(M, K, N, torch.float8_e4m3fn, n_iter=50)
        speedup = bf_us / fp_us
        print(f"({M}x{K})@({K}x{N})".ljust(24)
              + f"{bf_us:>10.1f} {bf_tfl:>10.1f} {fp_us:>10.1f} {fp_tfl:>10.1f} {speedup:>9.2f}x")

    print()
    print("=" * 72)
    print("FP8 quality vs FP32 reference (mean / max / p99 of |fp8 - fp32| / |fp32|)")
    print("=" * 72)
    for dist in ["normal", "softmax", "uniform"]:
        for shape in [(512, 512, 512), (2048, 2048, 2048)]:
            stats = quality_test(*shape, distribution=dist)
            M, K, N = shape
            print(f"{dist:<10} {M}x{K}@{K}x{N}  "
                  f"mean_rel_err={stats['mean_rel']:.4%}  "
                  f"p99_rel={stats['p99_rel']:.4%}  "
                  f"max_abs={stats['max_abs']:.4f}")


if __name__ == "__main__":
    main()
