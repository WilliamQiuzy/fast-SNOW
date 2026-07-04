"""For every SDPA call that has attn_mask != None, dump statistics about
the mask: is it all-zero (additive no-op)? all True (boolean no-op)?
some genuine mask? Group by (caller_site, shape) and report distribution.
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, traceback
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def caller_site():
    for fr in traceback.extract_stack()[::-1]:
        fn = fr.filename
        if ("analyze_attn_masks.py" in fn or "perflib/fa3.py" in fn or
            fn.endswith("torch/nn/functional.py") or fn.endswith("torch/overrides.py")):
            continue
        return f"{Path(fn).name}:{fr.lineno}"
    return "?"


def classify_mask(m):
    """Return a short label classifying the mask's effective content."""
    if m is None:
        return "None"
    if not isinstance(m, torch.Tensor):
        return f"non-tensor: {type(m).__name__}"
    if m.dtype == torch.bool:
        if m.all().item():
            return "bool-all-True (no-op)"
        if not m.any().item():
            return "bool-all-False (mask all)"
        true_frac = m.float().mean().item()
        return f"bool: {true_frac*100:.0f}% True"
    # numeric/additive mask
    nz = (m != 0).any().item()
    if not nz:
        return "additive-all-zero (no-op)"
    finite = torch.isfinite(m).all().item()
    has_neg_inf = (m == float("-inf")).any().item()
    if not finite and has_neg_inf:
        # additive masks: -inf where masked, 0 elsewhere
        masked_frac = (m == float("-inf")).float().mean().item()
        return f"additive -inf mask: {masked_frac*100:.0f}% masked"
    if finite:
        # finite values — could be a bias or low-mag mask
        absmax = m.abs().max().item()
        return f"additive finite (max|x|={absmax:.3f})"
    return f"additive mixed (neg-inf + finite)"


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.compile_memory_encoder = False
    cfg.sam3.compile_mask_decoder_transformer = False
    cfg.sam3.fa3_everywhere = False
    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    # Group by (caller_site, shape) — sample first occurrence's mask class.
    # Track distinct mask classifications per group too.
    import torch.nn.functional as F
    orig = F.scaled_dot_product_attention
    by_group: dict = defaultdict(lambda: {"count": 0, "classes": defaultdict(int)})
    def hook(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if attn_mask is not None and not is_causal:
            site = caller_site()
            shape_str = tuple(attn_mask.shape) if hasattr(attn_mask, "shape") else "?"
            key = (site, shape_str)
            by_group[key]["count"] += 1
            try:
                cls = classify_mask(attn_mask)
            except Exception as e:
                cls = f"classify-err: {type(e).__name__}"
            by_group[key]["classes"][cls] += 1
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
    F.scaled_dot_product_attention = hook
    import sys
    for modname in list(sys.modules):
        if "sam3" in modname:
            m = sys.modules[modname]
            if hasattr(m, "F") and hasattr(m.F, "scaled_dot_product_attention"):
                try: m.F.scaled_dot_product_attention = hook
                except Exception: pass

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]
    print("Warmup..."); pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
    by_group.clear()
    print("Measured run...")
    for v in videos:
        pool.run_inference(InferenceRequest(video_path=v, question=None))

    print("\n=== SDPA attn_mask classification ===")
    sorted_groups = sorted(by_group.items(), key=lambda x: -x[1]["count"])
    for (site, shape), info in sorted_groups:
        print(f"\n  {info['count']:>5} × {site}  shape={shape}")
        for cls, n in sorted(info["classes"].items(), key=lambda x: -x[1]):
            print(f"        {n:>5} × {cls}")
    total = sum(info["count"] for info in by_group.values())
    print(f"\nTotal masked SDPA calls: {total}")


if __name__ == "__main__":
    main()
