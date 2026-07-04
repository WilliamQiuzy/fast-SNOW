"""Set use_fa3=True on decoder Attention modules (currently False), then try
to compile. If FA3 path is preserved by AOTAutograd, this could be the win.
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return (len(n), sum(1 for x in n if x >= half), max(n, default=0))


def bench(pool, videos, label, runs=2):
    print(f"\n=== {label} ===")
    try:
        t0 = time.time()
        pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
        print(f"  warmup: {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  warmup err: {type(e).__name__}: {e}")
        return None
    avgs = []
    for k in range(runs):
        ts, q_all = [], []
        for v in videos:
            t0 = time.time()
            resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
            ts.append(time.time() - t0)
            sg = resp.four_dsg_dict if resp.status == "ok" else None
            q_all.append(quality(sg, pool.config.sampling.max_frames or 32) if sg else (0,0,0))
        avg = sum(ts) / 2
        avgs.append(avg)
        print(f"  run {k+1}: E1={ts[0]:.2f}s {q_all[0]}  E2={ts[1]:.2f}s {q_all[1]}  avg={avg:.2f}s")
    return sum(avgs) / len(avgs)


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.compile_memory_encoder = False
    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    dec = pool._sam3._predictor.model.tracker.model.sam_mask_decoder
    tfm = dec.transformer
    attn_modules = []
    for layer in tfm.layers:
        for name in ["self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"]:
            attn_modules.append(getattr(layer, name))
    if hasattr(tfm, "final_attn_token_to_image"):
        attn_modules.append(tfm.final_attn_token_to_image)

    bench(pool, videos, "baseline (use_fa3=False, no compile)")

    # 1. Turn on use_fa3 (no compile yet)
    for a in attn_modules: a.use_fa3 = True
    try:
        bench(pool, videos, "1. decoder use_fa3=True, no compile")
    except Exception as e:
        print(f"  FA3 path threw: {e}")

    # 2. use_fa3=True + transformer.compile
    tfm_orig = tfm.forward
    try:
        tfm.forward = torch.compile(tfm_orig, mode="reduce-overhead", dynamic=True)
        bench(pool, videos, "2. decoder use_fa3=True + transformer.compile")
    except Exception as e:
        print(f"  compile path threw: {e}")
    finally:
        tfm.forward = tfm_orig

    # 3. Use sdpa_kernel context to force flash attention in eager path
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for a in attn_modules: a.use_fa3 = False  # back to SDPA path

    # Patch Attention.forward to wrap SDPA in flash-only kernel context
    from sam3.sam.transformer import Attention as A_cls
    orig_attn_fwd = A_cls.forward
    def patched_forward(self, q, k, v):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return orig_attn_fwd(self, q, k, v)
    A_cls.forward = patched_forward
    try:
        bench(pool, videos, "3. SDPA forced flash (no compile)")
        tfm.forward = torch.compile(tfm_orig, mode="reduce-overhead", dynamic=True)
        bench(pool, videos, "4. SDPA forced flash + transformer.compile")
    except Exception as e:
        print(f"  flash-forced path threw: {e}")
    finally:
        tfm.forward = tfm_orig
        A_cls.forward = orig_attn_fwd


if __name__ == "__main__":
    main()
