"""V2: instrument the gate at model_misc.py:385 directly to see WHY it
falls back (attn_mask not None? is_causal? use_fa3 False?). Also walk the
stack further to find the actual caller module path.
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time, inspect
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.compile_memory_encoder = False
    cfg.sam3.compile_mask_decoder_transformer = False
    cfg.sam3.fa3_everywhere = True

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    import sam3.model.model_misc as mm
    orig_fmha = mm.functional_multi_head_attention if hasattr(mm, "functional_multi_head_attention") else None

    # Patch the inner functional attention to log why it fell back from FA3.
    # Easier path: just decorate the SDPA branch by hooking into F.scaled_dot_product_attention.
    import torch.nn.functional as F
    orig_sdpa = F.scaled_dot_product_attention
    # Reasons grouped by (caller_module_path, reason_string)
    fallback_reasons: Counter = Counter()
    fa3_counts: Counter = Counter()

    import sam3.perflib.fa3 as fa3_mod
    orig_flash = fa3_mod.flash_attn_func

    def walk_caller():
        """Return 'filename:lineno' for first frame outside instrumentation."""
        stack = inspect.stack()
        for fr in stack:
            fn = fr.filename
            if ("perflib/fa3.py" in fn or
                "verify_fa3_v2.py" in fn or
                fn.endswith("torch/nn/functional.py") or
                fn.endswith("torch/overrides.py")):
                continue
            return f"{Path(fn).name}:{fr.lineno}"
        return "?"

    def counting_flash(q, k, v):
        site = walk_caller()
        fa3_counts[site] += 1
        return orig_flash(q, k, v)
    fa3_mod.flash_attn_func = counting_flash

    def counting_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        # Find the caller and figure out why FA3 wasn't used
        site = walk_caller()
        reason = []
        if attn_mask is not None:
            reason.append(f"attn_mask={tuple(attn_mask.shape) if hasattr(attn_mask, 'shape') else type(attn_mask).__name__}")
        if is_causal:
            reason.append("is_causal=True")
        if not reason:
            reason.append("no-FA3-path")
        key = f"{site} :: {'+'.join(reason)}"
        fallback_reasons[key] += 1
        return orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
    F.scaled_dot_product_attention = counting_sdpa
    # Re-bind on module-level F imports that may have captured the original
    for modname in [
        "sam3.sam.transformer", "sam3.model.model_misc",
        "sam3.model.decoder", "sam3.model.vitdet",
    ]:
        try:
            mod = sys.modules.get(modname)
            if mod and hasattr(mod, "F"):
                mod.F.scaled_dot_product_attention = counting_sdpa
        except Exception:
            pass

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]
    print("Warmup..."); pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
    fa3_counts.clear(); fallback_reasons.clear()
    print("Measured run...")
    for v in videos:
        pool.run_inference(InferenceRequest(video_path=v, question=None))

    print("\n=== FA3 actual call sites ===")
    for site, n in sorted(fa3_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {n:>6} × {site}")
    print(f"  TOTAL FA3: {sum(fa3_counts.values())}")

    print("\n=== SDPA fallback (FA3 SKIPPED) by reason ===")
    for key, n in sorted(fallback_reasons.items(), key=lambda x: -x[1])[:25]:
        print(f"  {n:>6} × {key}")
    print(f"  TOTAL SDPA fallbacks: {sum(fallback_reasons.values())}")


if __name__ == "__main__":
    main()
