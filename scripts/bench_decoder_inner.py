"""Try compiling narrower submodules of the mask decoder to find a compile
target that preserves quality.

The full ``sam_mask_decoder.forward`` consistently breaks (all 0 tracks).
Suspect: Python-level branching on bool flags + dict mutation confuses dynamo
even with fullgraph=False. Try narrower targets:

  A. sam_mask_decoder.transformer.forward          (TwoWayTransformer)
  B. sam_mask_decoder.predict_masks                (inner method)
  C. sam_mask_decoder.transformer + output_upscaling (two sub-targets)
  D. sam_mask_decoder.iou_prediction_head.forward
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

import sys, time, types
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


def bench(pool, videos, label, runs=3):
    print(f"\n=== {label} ===")
    try:
        t0 = time.time()
        pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
        print(f"  warmup: {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  warmup err: {type(e).__name__}: {e}")
        return None, [(0,0,0)] * 2
    avgs, qs = [], []
    for k in range(runs):
        ts = []; q_all = []
        for v in videos:
            t0 = time.time()
            resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
            ts.append(time.time() - t0)
            sg = resp.four_dsg_dict if resp.status == "ok" else None
            q_all.append(quality(sg, pool.config.sampling.max_frames or 32) if sg else (0,0,0))
        avg = sum(ts) / 2
        avgs.append(avg); qs.append(q_all)
        print(f"  run {k+1}: E1={ts[0]:.2f}s {q_all[0]}  E2={ts[1]:.2f}s {q_all[1]}  avg={avg:.2f}s")
    mean = sum(avgs) / len(avgs)
    print(f"  → mean={mean:.2f}s")
    return mean, qs[-1]


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.compile_memory_encoder = False  # isolate the mask decoder

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    decoder = pool._sam3._predictor.model.tracker.model.sam_mask_decoder
    base_mean, base_q = bench(pool, videos, "baseline (no compile)")

    # A. Inner transformer (TwoWayTransformer) only
    tfm = decoder.transformer
    tfm_orig = tfm.forward
    try:
        tfm.forward = torch.compile(tfm_orig, mode="reduce-overhead", dynamic=True)
        bench(pool, videos, "A. decoder.transformer.compile (dyn=True)", runs=3)
    finally:
        tfm.forward = tfm_orig

    try:
        tfm.forward = torch.compile(tfm_orig, mode="reduce-overhead", dynamic=False)
        bench(pool, videos, "A2. decoder.transformer.compile (dyn=False)", runs=3)
    finally:
        tfm.forward = tfm_orig

    # B. predict_masks only
    pm_orig = decoder.predict_masks
    try:
        decoder.predict_masks = types.MethodType(
            torch.compile(pm_orig.__func__, dynamic=True), decoder)
        bench(pool, videos, "B. decoder.predict_masks.compile", runs=3)
    finally:
        decoder.predict_masks = pm_orig

    # C. transformer + output_upscaling Sequential
    ups = decoder.output_upscaling
    ups_orig = ups.forward
    try:
        tfm.forward = torch.compile(tfm_orig, mode="reduce-overhead", dynamic=True)
        ups.forward = torch.compile(ups_orig, mode="reduce-overhead", dynamic=True)
        bench(pool, videos, "C. transformer + output_upscaling compile", runs=3)
    finally:
        tfm.forward = tfm_orig
        ups.forward = ups_orig

    # D. iou_prediction_head only (tiny, just check correctness)
    iph = decoder.iou_prediction_head
    iph_orig = iph.forward
    try:
        iph.forward = torch.compile(iph_orig, dynamic=True)
        bench(pool, videos, "D. iou_prediction_head.compile", runs=2)
    finally:
        iph.forward = iph_orig


if __name__ == "__main__":
    main()
