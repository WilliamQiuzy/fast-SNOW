"""Measure ROSE pipeline throughput on VLM4D mini-200.

Target: 10 Hz (paper claim). Compute observed Hz per video, scatter over
num_tracks to answer: "how many objects can we track at 10 Hz?"

For each video:
  Hz_throughput = max_frames / total_inference_time

Models are warmed once on a throw-away first run; subsequent runs are
reported as "warm" (SAM3 stays loaded; DA3 reloads each call but FastSAM
stays warm too — the unload calls live inside _process_sam3_chunk).
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.pipeline.rose_e2e import ROSEEndToEnd

VLM4D_ROOT = ROOT / "benchmark" / "VLM4D-video"
HF_PREFIX = "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/"

# Benchmark config
N_VIDEOS = 15
MAX_FRAMES = 32        # matches paper's keyframe budget N_kf=32
TARGET_FPS = 10.0      # the 10 Hz target
USE_FA3 = True


def hf_url_to_local(url: str) -> Path:
    rel = url.replace(HF_PREFIX, "")
    return VLM4D_ROOT / rel


def make_config() -> ROSEConfig:
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sampling.max_frames = MAX_FRAMES
    cfg.sampling.target_fps = TARGET_FPS
    cfg.sam3.use_fa3 = USE_FA3
    # Default SAM3/perception thresholds (no tweaks)
    return cfg


def main():
    rng = random.Random(42)

    # Collect unique videos from mini-200
    videos = []
    for fname in ["mini_real_mc.json", "mini_synthetic_mc.json"]:
        with open(VLM4D_ROOT / "QA" / fname) as f:
            for q in json.load(f):
                p = hf_url_to_local(q["video"])
                if p.is_file() and p not in videos:
                    videos.append(p)

    print(f"Total unique mini videos found locally: {len(videos)}")
    sample = rng.sample(videos, min(N_VIDEOS, len(videos)))
    print(f"Benchmarking {len(sample)} videos at max_frames={MAX_FRAMES}, target_fps={TARGET_FPS}")
    print(f"FA3 enabled: {USE_FA3}")

    # Build e2e once (models load lazily during first call, then stay warm)
    e2e = ROSEEndToEnd(make_config())

    rows = []
    for i, vp in enumerate(sample, 1):
        warm = "WARM" if i > 1 else "COLD"
        try:
            t0 = time.time()
            res = e2e.build_4dsg_from_video(vp)
            dt = time.time() - t0
        except Exception as exc:
            print(f"[{i:>2}/{len(sample)}] {vp.name:35s}  FAILED: {type(exc).__name__}: {str(exc)[:100]}")
            continue

        md = res.four_dsg_dict.get("metadata", {})
        n_frames = md.get("num_frames", 0)
        n_tracks = md.get("num_tracks", 0)
        hz = n_frames / dt if dt > 0 else 0.0
        rows.append({
            "video": str(vp.relative_to(VLM4D_ROOT)),
            "warm": warm,
            "n_frames": n_frames,
            "n_tracks": n_tracks,
            "time_s": dt,
            "throughput_hz": hz,
        })
        print(f"[{i:>2}/{len(sample)}] {vp.name:35s}  {warm}  frames={n_frames:>2}  tracks={n_tracks:>2}  time={dt:5.1f}s  Hz={hz:5.2f}")
        res.cleanup()

    # Aggregate
    print("\n" + "=" * 80)
    print("  THROUGHPUT SUMMARY")
    print("=" * 80)
    warm_rows = [r for r in rows if r["warm"] == "WARM"]
    if warm_rows:
        hzs = [r["throughput_hz"] for r in warm_rows]
        ts = [r["n_tracks"] for r in warm_rows]
        print(f"Warm runs: {len(warm_rows)}")
        print(f"  Hz median:   {statistics.median(hzs):.2f}")
        print(f"  Hz mean:     {statistics.mean(hzs):.2f}")
        print(f"  Hz min/max:  {min(hzs):.2f} / {max(hzs):.2f}")
        print(f"  num_tracks median: {statistics.median(ts)}")
        print(f"  num_tracks min/max: {min(ts)} / {max(ts)}")

        # Hz vs num_tracks scatter
        print("\n  num_tracks  → median Hz")
        from collections import defaultdict
        by_t = defaultdict(list)
        for r in warm_rows:
            by_t[r["n_tracks"]].append(r["throughput_hz"])
        for t in sorted(by_t):
            hzs = by_t[t]
            print(f"    {t:>3}  ({len(hzs):>2} videos)  median Hz = {statistics.median(hzs):5.2f}  mean = {statistics.mean(hzs):5.2f}")

        # Find max tracks at 10 Hz threshold
        ten_hz_rows = [r for r in warm_rows if r["throughput_hz"] >= 10.0]
        below_rows = [r for r in warm_rows if r["throughput_hz"] < 10.0]
        if ten_hz_rows:
            max_t_at_10hz = max(r["n_tracks"] for r in ten_hz_rows)
            print(f"\n  ✓ Achieved >= 10 Hz on {len(ten_hz_rows)}/{len(warm_rows)} videos")
            print(f"    max num_tracks at 10 Hz: {max_t_at_10hz}")
        else:
            print(f"\n  ✗ NO video achieved 10 Hz (best: {max(r['throughput_hz'] for r in warm_rows):.2f} Hz)")
        if below_rows:
            print(f"    {len(below_rows)} videos < 10 Hz, num_tracks distribution: {sorted(set(r['n_tracks'] for r in below_rows))}")

    # Dump JSON for further analysis
    out = ROOT / "benchmark" / "throughput_mini200.json"
    with open(out, "w") as f:
        json.dump({"config": {"max_frames": MAX_FRAMES, "target_fps": TARGET_FPS, "use_fa3": USE_FA3},
                   "rows": rows}, f, indent=2, default=str)
    print(f"\nResults JSON: {out}")


if __name__ == "__main__":
    main()
