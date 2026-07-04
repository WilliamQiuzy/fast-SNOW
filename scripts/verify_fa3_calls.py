"""Runtime check: count every actual call to flash_attn_func and group by
its Python call site. Compares which use_fa3=True modules ACTUALLY invoke
FA3 vs which silently fall through to SDPA.
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time, traceback
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
    cfg.sam3.compile_memory_encoder = False  # avoid hiding call sites
    cfg.sam3.compile_mask_decoder_transformer = False
    cfg.sam3.fa3_everywhere = True  # FLIP all to True to see what actually fires

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    # Count FA3 invocations by Python caller (file:line).
    fa3_call_sites: Counter = Counter()
    import sam3.perflib.fa3 as fa3_mod
    orig_flash = fa3_mod.flash_attn_func
    def counting_flash(q, k, v):
        # Walk stack — find the first frame OUTSIDE sam3.perflib.fa3
        for fr in traceback.extract_stack()[::-1]:
            if "perflib/fa3.py" not in fr.filename:
                key = f"{Path(fr.filename).name}:{fr.lineno}"
                fa3_call_sites[key] += 1
                break
        return orig_flash(q, k, v)
    fa3_mod.flash_attn_func = counting_flash

    # Also count SDPA fallbacks in attention paths we suspect
    sdpa_call_sites: Counter = Counter()
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def counting_sdpa(*args, **kwargs):
        for fr in traceback.extract_stack()[::-1]:
            fn = fr.filename
            # Only count call sites inside sam3 source — ignore torch internals
            if "sam3/" in fn and "perflib" not in fn:
                key = f"{Path(fn).name}:{fr.lineno}"
                sdpa_call_sites[key] += 1
                break
        return orig_sdpa(*args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = counting_sdpa

    # Patch the F.scaled_dot_product_attention in each module that imports it directly
    # (because they did `from torch.nn import functional as F` and bound it locally).
    import sam3.sam.transformer as tfm_mod
    import sam3.model.model_misc as mm_mod
    import sam3.model.decoder as dec_mod
    import sam3.model.vitdet as vit_mod
    for mod in (tfm_mod, mm_mod, dec_mod, vit_mod):
        if hasattr(mod, "F"):
            try:
                mod.F.scaled_dot_product_attention = counting_sdpa
            except Exception:
                pass

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    print("\nRun 1 (warmup) ...")
    pool.run_inference(InferenceRequest(video_path=videos[0], question=None))

    # Reset counters after warmup
    fa3_call_sites.clear()
    sdpa_call_sites.clear()

    print("Run 2 (measured) ...")
    for v in videos:
        pool.run_inference(InferenceRequest(video_path=v, question=None))

    print("\n=== FA3 call sites (counted across 2 videos) ===")
    for site, n in sorted(fa3_call_sites.items(), key=lambda x: -x[1]):
        print(f"  {n:>6} × {site}")
    print(f"  TOTAL FA3 calls: {sum(fa3_call_sites.values())}")

    print("\n=== SDPA call sites (NOT going through FA3) ===")
    for site, n in sorted(sdpa_call_sites.items(), key=lambda x: -x[1]):
        print(f"  {n:>6} × {site}")
    print(f"  TOTAL SDPA calls: {sum(sdpa_call_sites.values())}")


if __name__ == "__main__":
    main()
