"""K-vs-time on a series of synthetic dense scenes K in {5,10,20,30,40,50}.

Each video is 150 frames @ 30 fps (5 s) with N moving colored disks on a
near-black background.  Disks are spatially distinct so FastSAM detects each
as one mask, multiplex tracking yields exactly N tracks (modulo a few
over-segmentation duplicates that get dedup'd).

Output:
  - benchmark/k_tracking.json  raw timings
  - benchmark/k_tracking.pdf   K vs construction time / throughput
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

VIDEO_DIR = Path("/tmp/synth_dense")
TARGET_KS = [5, 10, 20, 30, 40, 50]
CONF_THRESHOLD = 0.40  # strikes balance between over-detection and missing
N_TRIALS = 2
OUT_DIR = ROOT / "benchmark"


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = True
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = 80
    cfg.sam3.num_maskmem = 7
    cfg.sampling.max_frames = None
    cfg.sampling.target_fps = 10.0
    cfg.fastsam.conf_threshold = CONF_THRESHOLD

    print("Loading warm pool (compile=True)...", flush=True)
    t0 = time.time()
    pool = WarmModelPool(cfg)
    pool.load_all()
    pool.warmup_cuda()
    pool._status = "ready"
    print(f"  pool ready in {time.time() - t0:.1f}s", flush=True)

    # One global warmup at the densest video so compile sees the largest
    # tensor shapes upfront.
    big_video = VIDEO_DIR / f"synth_dense_K{TARGET_KS[-1]:02d}.mp4"
    print(f"  global warmup on {big_video.name}...", flush=True)
    t = time.time()
    pool.run_inference(InferenceRequest(video_path=str(big_video), question=None))
    print(f"    warmup done in {time.time() - t:.1f}s", flush=True)

    rows = []
    for K_target in TARGET_KS:
        video = VIDEO_DIR / f"synth_dense_K{K_target:02d}.mp4"
        if not video.is_file():
            print(f"  MISSING: {video}", flush=True)
            continue

        ts, K_obs, n_frames = [], [], 0
        for trial in range(N_TRIALS):
            t = time.time()
            resp = pool.run_inference(InferenceRequest(
                video_path=str(video), question=None,
            ))
            dt = time.time() - t
            if resp.status != "ok":
                print(f"  K_target={K_target}: FAILED {resp.error_message}", flush=True)
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
        row = dict(K_target=K_target, K=med_K, frames=n_frames,
                   trials=ts, median_t=med_t, hz=hz,
                   conf_threshold=CONF_THRESHOLD)
        rows.append(row)
        print(f"  K_target={K_target:>2}  K_obs={med_K:>2}  median={med_t:5.2f}s  Hz={hz:5.2f}",
              flush=True)

        (OUT_DIR / "k_tracking.json").write_text(json.dumps({
            "_note": (f"K-tracking sweep on synthetic dense disks (5s @ 30 fps). "
                      f"FastSAM threshold fixed at {CONF_THRESHOLD}; varying scene "
                      f"K from {TARGET_KS[0]} to {TARGET_KS[-1]} disks."),
            "videos_dir": str(VIDEO_DIR),
            "config": {
                "max_active_tracks": cfg.sam3.max_active_tracks,
                "target_fps": cfg.sampling.target_fps,
                "max_frames": cfg.sampling.max_frames,
                "anchor_stride": cfg.sam3.anchor_stride,
                "fastsam_conf_threshold": CONF_THRESHOLD,
            },
            "rows": rows,
        }, indent=2))

    print("\nFinal results:", flush=True)
    for r in rows:
        print(f"  K_target={r['K_target']:>2}  K_obs={r['K']:>2}  "
              f"t={r['median_t']:.2f}s  Hz={r['hz']:.2f}", flush=True)
    print(f"\nSaved {OUT_DIR / 'k_tracking.json'}", flush=True)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rs = sorted(rows, key=lambda r: r["K"])
        Ks = [r["K"] for r in rs]
        ts = [r["median_t"] for r in rs]
        hzs = [r["hz"] for r in rs]
        n_frames = rs[0]["frames"]

        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
        axes[0].plot(Ks, ts, marker="o", color="C0", linewidth=2)
        axes[0].set_xlabel("Tracked objects $K$")
        axes[0].set_ylabel("4DSG construction time (s)")
        axes[0].set_title(f"$K$ vs construction time\n(synthetic dense, "
                          f"{n_frames} frames @ 10 fps)")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(0, max(Ks) + 5, 5))

        axes[1].plot(Ks, hzs, marker="s", color="C1", linewidth=2,
                     label="ROSE (this work)")
        axes[1].axhline(10, color="grey", linestyle=":", linewidth=1.2,
                         label="paper claim 10 fps")
        axes[1].set_xlabel("Tracked objects $K$")
        axes[1].set_ylabel("Throughput (fps)")
        axes[1].set_title("$K$ vs throughput")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        axes[1].set_xticks(range(0, max(Ks) + 5, 5))

        out_pdf = OUT_DIR / "k_tracking.pdf"
        out_png = OUT_DIR / "k_tracking.png"
        fig.savefig(str(out_pdf), bbox_inches="tight")
        fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
        print(f"Saved {out_pdf}, {out_png}", flush=True)
    except Exception as e:
        print(f"Plot failed (non-fatal): {e}", flush=True)


if __name__ == "__main__":
    main()
