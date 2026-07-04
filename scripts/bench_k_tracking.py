"""K-vs-time curve on a SINGLE video by sweeping FastSAM threshold.

Setup:
  - One video, paper-faithful sampling (target_fps=10, no max_frames cap).
  - Vary `fastsam.conf_threshold` to control how many unique objects
    FastSAM proposes; SAM3.1 multiplex tracks all of them.  Higher
    threshold → fewer detections → smaller K.  Lower threshold → more
    detections → larger K.
  - Measure per-video wall clock at each K (1 warmup + 2 timed trials).

Output:
  - benchmark/k_tracking.json     raw data
  - benchmark/k_tracking.pdf/.png matplotlib curve (K vs time, K vs Hz)
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest

# Use a crowded DSR-Bench street scene.  yWgmyNBbZ-E_9 is 18 s × 30 fps,
# yields 180 frames at our 10 fps sampling; frame 0 has 111 FastSAM
# detections at conf=0.10 — dense enough to probe K up to ~50 after
# cross-anchor dedup, while keeping per-trial wall ~30-60 s instead of
# the 100+ s a 42 s video would take.
PROBE_VIDEO = (
    ROOT / "benchmark" / "DSR-Bench" / "videos"
    / "part_01" / "yWgmyNBbZ-E_9.mp4"
)

# Threshold sweep — chosen so that resulting K spans roughly 5..50.
# Lower conf threshold → FastSAM keeps more detections.
# Frame-0 FastSAM detections at these thresholds for yWgmyNBbZ-E_9:
#   0.55→3, 0.40→9, 0.25→27, 0.15→62, 0.10→111, 0.05→200, 0.02→200, 0.01→200
THRESHOLDS = [0.55, 0.40, 0.25, 0.15, 0.10, 0.05, 0.02, 0.01]

OUT_DIR = ROOT / "benchmark"


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = True
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = 80   # high enough to never bind for K≤50
    cfg.sam3.num_maskmem = 7
    cfg.sampling.max_frames = None    # paper-faithful
    cfg.sampling.target_fps = 10.0

    print("Loading warm pool (compile=True)...", flush=True)
    t0 = time.time()
    pool = WarmModelPool(cfg)
    pool.load_all()
    pool.warmup_cuda()
    pool._status = "ready"
    print(f"  pool ready in {time.time() - t0:.1f}s", flush=True)

    if not PROBE_VIDEO.is_file():
        print(f"  MISSING video: {PROBE_VIDEO}", flush=True)
        return

    print(f"\nProbe video: {PROBE_VIDEO.name}", flush=True)

    # One global warmup at the densest threshold to compile every shape we'll
    # see — subsequent threshold sweeps reuse the cached compile graphs.
    pool.config.fastsam.conf_threshold = float(THRESHOLDS[-1])
    pool._fastsam.config.conf_threshold = float(THRESHOLDS[-1])
    print("  global warmup (at lowest threshold)...", flush=True)
    t = time.time()
    pool.run_inference(InferenceRequest(video_path=str(PROBE_VIDEO),
                                        question=None))
    print(f"    warmup done in {time.time() - t:.1f}s", flush=True)

    rows = []

    for thr in THRESHOLDS:
        # Mutate FastSAM threshold — re-loaded model picks it up in detect().
        pool.config.fastsam.conf_threshold = float(thr)
        pool._fastsam.config.conf_threshold = float(thr)
        # Timed trials (2).
        ts = []
        K_obs = []
        n_frames = 0
        for _ in range(2):
            t = time.time()
            resp = pool.run_inference(InferenceRequest(
                video_path=str(PROBE_VIDEO), question=None,
            ))
            dt = time.time() - t
            if resp.status != "ok":
                print(f"  conf={thr}: FAILED {resp.error_message}", flush=True)
                break
            ts.append(dt)
            md = (resp.four_dsg_dict or {}).get("metadata", {})
            n_frames = md.get("num_frames", 0)
            K_obs.append(md.get("num_tracks", 0))
        if not ts:
            continue
        med_t = statistics.median(ts)
        med_K = int(statistics.median(K_obs))
        hz = n_frames / med_t if med_t > 0 else 0
        row = dict(conf_threshold=thr, K=med_K, frames=n_frames,
                   trials=ts, median_t=med_t, hz=hz)
        rows.append(row)
        print(f"  conf={thr:>5.2f}  K={med_K:>2}  median={med_t:5.2f}s  Hz={hz:5.2f}",
              flush=True)

        # Persist after each row so we don't lose data on crash/kill.
        (OUT_DIR / "k_tracking.json").write_text(json.dumps({
            "video": str(PROBE_VIDEO),
            "config": {
                "max_active_tracks": cfg.sam3.max_active_tracks,
                "target_fps": cfg.sampling.target_fps,
                "max_frames": cfg.sampling.max_frames,
                "anchor_stride": cfg.sam3.anchor_stride,
            },
            "rows": rows,
        }, indent=2))

    # Save raw
    out_json = OUT_DIR / "k_tracking.json"
    out_json.write_text(json.dumps({
        "video": str(PROBE_VIDEO),
        "config": {
            "max_active_tracks": cfg.sam3.max_active_tracks,
            "target_fps": cfg.sampling.target_fps,
            "max_frames": cfg.sampling.max_frames,
            "anchor_stride": cfg.sam3.anchor_stride,
        },
        "rows": rows,
    }, indent=2))
    print(f"\nSaved {out_json}", flush=True)

    # Plot K vs time and K vs Hz
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows_sorted = sorted(rows, key=lambda r: r["K"])
        Ks = [r["K"] for r in rows_sorted]
        ts = [r["median_t"] for r in rows_sorted]
        hzs = [r["hz"] for r in rows_sorted]
        thrs = [r["conf_threshold"] for r in rows_sorted]

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)

        axes[0].plot(Ks, ts, marker="o", color="C0")
        for x, y, t in zip(Ks, ts, thrs):
            axes[0].annotate(f"τ={t}", (x, y), fontsize=6,
                              textcoords="offset points", xytext=(4, 4))
        axes[0].set_xlabel("Tracked objects $K$")
        axes[0].set_ylabel("4DSG construction time (s)")
        axes[0].set_title(f"$K$ vs. construction time\n({PROBE_VIDEO.name}, "
                          f"{n_frames} frames @ 10 fps)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(Ks, hzs, marker="s", color="C1")
        for x, y, t in zip(Ks, hzs, thrs):
            axes[1].annotate(f"τ={t}", (x, y), fontsize=6,
                              textcoords="offset points", xytext=(4, 4))
        axes[1].set_xlabel("Tracked objects $K$")
        axes[1].set_ylabel("Throughput (fps)")
        axes[1].set_title("$K$ vs. throughput")
        axes[1].axhline(10, color="grey", linestyle=":", linewidth=1,
                         label="paper 10 fps")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7)

        out_pdf = OUT_DIR / "k_tracking.pdf"
        out_png = OUT_DIR / "k_tracking.png"
        fig.savefig(str(out_pdf), bbox_inches="tight")
        fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
        print(f"Saved {out_pdf}, {out_png}", flush=True)
    except Exception as e:
        print(f"Plot failed (non-fatal): {e}", flush=True)


if __name__ == "__main__":
    main()
