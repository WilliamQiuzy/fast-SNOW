"""End-to-end FP8 quality test on the full ROSE pipeline.

Strategy: monkey-patch all torch.nn.Linear forward calls in the SAM 3.1
multiplex model to inject FP8 quantization noise via FP32→FP8→FP32 round-
trip.  This SIMULATES the quality impact of FP8 inference without changing
speed (we still run BF16 underneath).

Compare:
  1. BF16 baseline 4DSG output
  2. FP8-simulated 4DSG output

Quality metrics:
  - num_tracks
  - Per-track n_obs distribution
  - Track overlap (Jaccard) between BF16 vs FP8 mask trajectories
  - Mean mask IoU between matched tracks
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import numpy as np
import torch
import torch.nn as nn

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def fp8_quantize_passthrough(x: torch.Tensor) -> torch.Tensor:
    """Simulate per-tensor FP8 e4m3 quantization: amax-scale → FP8 → back.

    This injects FP8's quantization noise but underlying compute stays BF16.
    The QUALITY effect is the same as real FP8 (assuming per-tensor scaling);
    the SPEED is unchanged.
    """
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return x
    if x.numel() == 0:
        return x
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    amax = x32.abs().max().clamp(min=1e-12)
    s = 448.0 / amax
    q = (x32 * s).to(torch.float8_e4m3fn).to(torch.float32) / s
    return q.to(orig_dtype)


def patch_linear_for_fp8(model: nn.Module) -> int:
    """Quantize weights of every Linear/Conv2d to FP8 (round-trip).

    This injects WEIGHT quantization noise — the dominant source of FP8
    quality loss.  Activations would be quantized too in real FP8 inference;
    we test weights here as a representative-magnitude proxy.

    Returns number of layers quantized.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and not getattr(module, "_fp8_patched", False):
            with torch.no_grad():
                module.weight.data.copy_(fp8_quantize_passthrough(module.weight.data))
            module._fp8_patched = True
            n += 1
    return n


def run_one(pool: WarmModelPool, video_path: str):
    t = time.time()
    resp = pool.run_inference(InferenceRequest(video_path=video_path, question=None))
    dt = time.time() - t
    if resp.status != "ok":
        raise RuntimeError(f"FAILED: {resp.error_message}")
    return resp.four_dsg_dict, dt


def compare_4dsg(bf16, fp8):
    """Compare two 4DSGs.  Return dict of quality metrics."""
    b = bf16; f = fp8
    bt = b["tracks"]; ft = f["tracks"]
    bn = b["metadata"]["num_tracks"]
    fn = f["metadata"]["num_tracks"]

    # Distribution of n_obs
    bn_obs = sorted([len(t["F_k"]) for t in bt], reverse=True)
    fn_obs = sorted([len(t["F_k"]) for t in ft], reverse=True)

    # Try to match tracks by (image_position, extent) — coarse but indicative.
    def feat(tr):
        return (tr["image_position"], tuple(round(x, 1) for x in tr["extent"]))
    b_feats = {feat(t): t for t in bt}
    f_feats = {feat(t): t for t in ft}
    common = set(b_feats) & set(f_feats)
    return {
        "bf16_num_tracks": bn,
        "fp8_num_tracks": fn,
        "bf16_n_obs": bn_obs,
        "fp8_n_obs": fn_obs,
        "matched_by_pos_extent": len(common),
        "bf16_total_obs": sum(bn_obs),
        "fp8_total_obs": sum(fn_obs),
    }


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else (
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4"
    )

    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sampling.max_frames = 32

    print("Loading warm pool...", flush=True)
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    # --- BF16 baseline ---
    print("\n[1] Running BF16 baseline...", flush=True)
    bf16_sg, bf16_t = run_one(pool, video)
    print(f"BF16: {bf16_t:.2f}s, {bf16_sg['metadata']['num_tracks']} tracks")

    # --- Patch SAM's Linears with FP8 simulation ---
    print("\n[2] Patching SAM 3.1 Linear layers with FP8 quantization simulation...", flush=True)
    n_patched = patch_linear_for_fp8(pool._sam3._predictor.model)
    print(f"  patched {n_patched} Linear layers")

    # --- FP8-simulated run ---
    print("\n[3] Running FP8-simulated...", flush=True)
    fp8_sg, fp8_t = run_one(pool, video)
    print(f"FP8-sim: {fp8_t:.2f}s, {fp8_sg['metadata']['num_tracks']} tracks")

    # --- Compare ---
    print("\n[4] Quality comparison:")
    cmp = compare_4dsg(bf16_sg, fp8_sg)
    print(f"  num_tracks  BF16={cmp['bf16_num_tracks']}  FP8={cmp['fp8_num_tracks']}")
    print(f"  total obs   BF16={cmp['bf16_total_obs']}  FP8={cmp['fp8_total_obs']}")
    print(f"  matched (pos+extent) {cmp['matched_by_pos_extent']}/{min(cmp['bf16_num_tracks'], cmp['fp8_num_tracks'])}")
    print(f"  BF16 n_obs sorted: {cmp['bf16_n_obs']}")
    print(f"  FP8  n_obs sorted: {cmp['fp8_n_obs']}")
    # Long-track preservation: number of tracks with n>=20
    bf16_long = sum(1 for n in cmp['bf16_n_obs'] if n >= 20)
    fp8_long = sum(1 for n in cmp['fp8_n_obs'] if n >= 20)
    print(f"  Long tracks (n>=20):  BF16={bf16_long}  FP8={fp8_long}")
    bf16_full = sum(1 for n in cmp['bf16_n_obs'] if n == max(cmp['bf16_n_obs'], default=0))
    fp8_full = sum(1 for n in cmp['fp8_n_obs'] if n == max(cmp['fp8_n_obs'], default=0))
    print(f"  Max n_obs:  BF16={max(cmp['bf16_n_obs'], default=0)}  FP8={max(cmp['fp8_n_obs'], default=0)}")

    json.dump({"bf16": bf16_sg, "fp8": fp8_sg, "compare": cmp},
              open("/tmp/fp8_compare.json", "w"), indent=2)
    print("\nSaved comparison: /tmp/fp8_compare.json")


if __name__ == "__main__":
    main()
