"""Final decoder compile attempts:

  - Compile output_upscaling only (no attention)
  - Compile self_attn ONLY (no cross_attn) for each layer
  - Try backend='eager' to isolate inductor vs dynamo tracing
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
    avgs, qs = [], []
    for k in range(runs):
        ts, q_all = [], []
        for v in videos:
            t0 = time.time()
            resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
            ts.append(time.time() - t0)
            sg = resp.four_dsg_dict if resp.status == "ok" else None
            q_all.append(quality(sg, pool.config.sampling.max_frames or 32) if sg else (0,0,0))
        avg = sum(ts) / 2
        avgs.append(avg); qs.append(q_all)
        print(f"  run {k+1}: E1={ts[0]:.2f}s {q_all[0]}  E2={ts[1]:.2f}s {q_all[1]}  avg={avg:.2f}s")
    return sum(avgs)/len(avgs), qs[-1]


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

    bench(pool, videos, "baseline")

    # 1. Compile output_upscaling alone (no attention)
    ups = dec.output_upscaling
    ups_orig = ups.forward
    try:
        ups.forward = torch.compile(ups_orig, mode="reduce-overhead", dynamic=True)
        bench(pool, videos, "1. output_upscaling.compile")
    finally:
        ups.forward = ups_orig

    # 2. backend="eager" on transformer (dynamo trace, no inductor)
    tfm_orig = tfm.forward
    try:
        tfm.forward = torch.compile(tfm_orig, backend="eager", dynamic=True)
        bench(pool, videos, "2. transformer backend=eager")
    finally:
        tfm.forward = tfm_orig

    # 3. backend="aot_eager" — adds AOTAutograd but no inductor codegen
    try:
        tfm.forward = torch.compile(tfm_orig, backend="aot_eager", dynamic=True)
        bench(pool, videos, "3. transformer backend=aot_eager")
    finally:
        tfm.forward = tfm_orig

    # 4. Compile ONLY layer[i].self_attn instead of whole layer
    for li in range(len(tfm.layers)):
        for name in ["self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"]:
            attn = getattr(tfm.layers[li], name)
            attn_orig = attn.forward
            try:
                attn.forward = torch.compile(attn_orig, mode="reduce-overhead", dynamic=True)
                bench(pool, videos, f"4. tfm.layers[{li}].{name}.compile")
            finally:
                attn.forward = attn_orig


if __name__ == "__main__":
    main()
