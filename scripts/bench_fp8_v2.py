"""FP8 quality test with PROPER per-row scaling (as TransformerEngine does).

The naive 'one scale per matrix' approach gives 40-80% error.  Real FP8
deployments use PER-ROW or PER-TILE scaling so each row's amax fits the FP8
dynamic range exactly.  This gives the best-case FP8 quality.
"""
import torch

device = torch.device("cuda")


def fp8_matmul_rowscale(a_fp32: torch.Tensor, b_fp32: torch.Tensor) -> torch.Tensor:
    """A @ B with per-row scaling of A and per-column scaling of B (the TE way).

    Falls back to per-tensor scale when _scaled_mm doesn't support per-row in
    this torch version (we expect it to require manual fallback).
    """
    M, K = a_fp32.shape
    K2, N = b_fp32.shape
    assert K == K2

    # Per-row amax of A (M,), per-col amax of B (N,)
    a_amax = a_fp32.abs().amax(dim=1).clamp(min=1e-12)        # (M,)
    b_amax = b_fp32.abs().amax(dim=0).clamp(min=1e-12)        # (N,)
    s_a = (448.0 / a_amax).to(torch.float32)                  # scale to FP8 max
    s_b = (448.0 / b_amax).to(torch.float32)
    a_scaled = (a_fp32 * s_a[:, None]).to(torch.float8_e4m3fn)
    b_scaled = (b_fp32 * s_b[None, :]).to(torch.float8_e4m3fn)
    # Now matmul in FP8 (a_scaled @ b_scaled) gives integer-domain product.
    # We rescale by (1/s_a)(1/s_b) per (row, col) of output.
    b_T = b_scaled.t().contiguous().t()  # column-major
    # _scaled_mm uses a single scale → use per-tensor fallback for now, then
    # apply per-row/col correction in FP32 after.
    scale_a = torch.tensor([1.0], device=device)
    scale_b = torch.tensor([1.0], device=device)
    out_int, _ = torch._scaled_mm(a_scaled, b_T, scale_a=scale_a, scale_b=scale_b,
                                   out_dtype=torch.float32)
    # Apply per-row/col scale undo: out[i,j] = out_int[i,j] / (s_a[i] * s_b[j])
    out = out_int / (s_a[:, None] * s_b[None, :])
    return out


def quality_compare(M, K, N, dist="normal", per_row=True):
    if dist == "normal":
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
    elif dist == "softmax":
        a = torch.softmax(torch.randn(M, K, device=device), dim=-1).float()
        b = torch.randn(K, N, device=device, dtype=torch.float32) * 0.1
    elif dist == "attention_like":
        # Like attention output: Q @ K^T softmax @ V
        q = torch.randn(M, K, device=device, dtype=torch.float32) * 0.5
        k = torch.randn(N, K, device=device, dtype=torch.float32) * 0.5
        v = torch.randn(N, N, device=device, dtype=torch.float32) * 0.5
        a = torch.softmax(q @ k.t() / (K ** 0.5), dim=-1)
        b = v

    ref = a @ b

    if per_row:
        out_fp8 = fp8_matmul_rowscale(a, b)
    else:
        # Per-tensor scaling
        s_a = 448.0 / a.abs().max().clamp(min=1e-12)
        s_b = 448.0 / b.abs().max().clamp(min=1e-12)
        a_fp8 = (a * s_a).to(torch.float8_e4m3fn)
        b_fp8 = (b * s_b).to(torch.float8_e4m3fn)
        scale_a = torch.tensor([1.0 / s_a.item()], device=device)
        scale_b = torch.tensor([1.0 / s_b.item()], device=device)
        out_fp8, _ = torch._scaled_mm(a_fp8, b_fp8.t().contiguous().t(),
                                       scale_a=scale_a, scale_b=scale_b,
                                       out_dtype=torch.float32)

    abs_err = (out_fp8 - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-6)
    return rel_err.mean().item(), rel_err.flatten().quantile(0.99).item()


def main():
    print("=" * 80)
    print("FP8 QUALITY — per-row scaling (TE-style) vs per-tensor (naive)")
    print("=" * 80)
    print(f"{'distribution':<18} {'shape':<22} {'PER-TENSOR mean/p99':<28} {'PER-ROW mean/p99'}")
    for dist in ["normal", "softmax", "attention_like"]:
        for shape in [(512, 512, 512), (2048, 2048, 2048), (4096, 128, 4096)]:
            M, K, N = shape
            try:
                m1, p1 = quality_compare(M, K, N, dist=dist, per_row=False)
                m2, p2 = quality_compare(M, K, N, dist=dist, per_row=True)
                print(f"{dist:<18} {M}x{K}@{K}x{N}     "
                      f"{m1*100:6.3f}% / {p1*100:6.2f}%       "
                      f"{m2*100:6.3f}% / {p2*100:6.2f}%")
            except Exception as e:
                print(f"{dist:<18} {shape}   FAIL: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
