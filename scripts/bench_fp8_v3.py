"""FP8 quality test — ABSOLUTE error vs output magnitude.

Relative error explodes near zero outputs (e.g. random matmul outputs).
What actually matters for downstream quality:
  - mean( |out_fp8 - out_fp32| ) / std(out_fp32)   (SNR-like)
  - For attention: does softmax(QK^T fp8) ≈ softmax(QK^T fp32)?
"""
import torch
import torch.nn.functional as F

device = torch.device("cuda")


def fp8_matmul(a_fp32, b_fp32):
    s_a = 448.0 / a_fp32.abs().max().clamp(min=1e-12)
    s_b = 448.0 / b_fp32.abs().max().clamp(min=1e-12)
    a_fp8 = (a_fp32 * s_a).to(torch.float8_e4m3fn)
    b_fp8 = (b_fp32 * s_b).to(torch.float8_e4m3fn)
    scale_a = torch.tensor([1.0 / s_a.item()], device=device)
    scale_b = torch.tensor([1.0 / s_b.item()], device=device)
    out, _ = torch._scaled_mm(a_fp8, b_fp8.t().contiguous().t(),
                              scale_a=scale_a, scale_b=scale_b,
                              out_dtype=torch.float32)
    return out


def attention_quality(B, S, D):
    """Simulate self-attention.  Q,K,V are (B,S,D); output is (B,S,D).

    Compare BF16 vs FP8 paths:
      QK^T → softmax → attention @ V
    """
    q = torch.randn(B, S, D, device=device) * 0.1
    k = torch.randn(B, S, D, device=device) * 0.1
    v = torch.randn(B, S, D, device=device) * 0.1

    # FP32 reference
    scores_ref = q @ k.transpose(-1, -2) / (D ** 0.5)
    attn_ref = F.softmax(scores_ref, dim=-1)
    out_ref = attn_ref @ v   # (B, S, D)

    # FP8 path: QK^T in FP8, softmax in FP32 (TE keeps softmax in higher prec),
    # AV in FP8.
    out_fp8 = torch.zeros_like(out_ref)
    for bi in range(B):
        scores_fp8 = fp8_matmul(q[bi], k[bi].t().contiguous().t().to(torch.float32))
        scores_fp8 = scores_fp8 / (D ** 0.5)
        attn_fp8 = F.softmax(scores_fp8, dim=-1)
        out_fp8[bi] = fp8_matmul(attn_fp8, v[bi])

    # Stats
    out_diff = (out_fp8 - out_ref)
    out_std = out_ref.std().item()
    out_max = out_ref.abs().max().item()
    snr_db = 20 * torch.log10(out_ref.std() / (out_diff.std() + 1e-12))
    # Compare attention WEIGHTS (the softmax output) since that's what matters
    # for "where the model looks"
    attn_diff = (attn_fp8 - attn_ref) if False else None  # placeholder

    print(f"  attention B={B} S={S} D={D}:")
    print(f"    out FP32 std={out_std:.4f}  max={out_max:.4f}")
    print(f"    diff (FP8 - FP32) mean_abs={out_diff.abs().mean():.5f}  "
          f"max_abs={out_diff.abs().max():.5f}")
    print(f"    SNR(out): {snr_db.item():.1f} dB  "
          f"(higher = better; 40dB ≈ 1% noise, 60dB ≈ 0.1%)")


def linear_quality(in_features, out_features, batch_size):
    """Linear layer (W @ x + b) FP8 vs FP32."""
    W = torch.randn(in_features, out_features, device=device) * (1 / in_features ** 0.5)
    x = torch.randn(batch_size, in_features, device=device)
    ref = x @ W
    fp8 = fp8_matmul(x, W)
    diff = (fp8 - ref)
    snr_db = 20 * torch.log10(ref.std() / (diff.std() + 1e-12))
    print(f"  Linear in={in_features} out={out_features} batch={batch_size}:")
    print(f"    SNR={snr_db.item():.1f} dB  "
          f"mean_abs={diff.abs().mean():.5f}/{ref.abs().mean():.5f}={diff.abs().mean()/ref.abs().mean()*100:.2f}%")


def main():
    print("=" * 80)
    print("FP8 quality on REALISTIC neural-net workloads (SAM-attention sized)")
    print("=" * 80)
    print()
    print("Linear layer SNR (rule of thumb: > 40 dB OK; > 60 dB excellent):")
    for cfg in [(256, 256, 256), (512, 512, 1024), (1024, 1024, 1024), (4096, 4096, 1024)]:
        linear_quality(*cfg)
    print()
    print("Attention SNR (with proper softmax in FP32):")
    for B, S, D in [(1, 256, 64), (1, 1024, 64), (4, 512, 128), (8, 1024, 128)]:
        attention_quality(B, S, D)


if __name__ == "__main__":
    main()
