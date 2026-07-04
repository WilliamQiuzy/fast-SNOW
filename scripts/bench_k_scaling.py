"""Sweep max_active_tracks K and measure per-video inference time.

Reuses one warm pool across all K values to avoid paying cold-compile cost
per K.  For each K, runs the same video 3 times (1 warmup + 2 measure) and
records the median inference time.

Outputs:
  - benchmark/k_scaling.json   raw data
  - benchmark/k_scaling.pdf    matplotlib plot (Hz vs K)
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

VLM4D = ROOT / "benchmark" / "VLM4D-video"
OUT_DIR = ROOT / "benchmark"
K_VALUES = [5, 10, 15, 20, 30, 40, 50]
# Pick crowded scenes — synth videos typically have 10-14 objects.
PROBE_VIDEOS = [
    "videos_synthetic/synth_350.mp4",  # ~14 tracks
    "videos_synthetic/synth_241.mp4",  # ~10 tracks
    "videos_synthetic/synth_310.mp4",  # ~10 tracks
]
N_TRIALS_WARMUP = 1
N_TRIALS_MEASURE = 2


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = True   # v45 final config
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = max(K_VALUES)  # initial; mutated per-K below
    cfg.sam3.num_maskmem = 7
    cfg.sampling.max_frames = 32
    cfg.sampling.target_fps = 10.0

    print("Loading warm pool (compile=True)...", flush=True)
    t0 = time.time()
    pool = WarmModelPool(cfg)
    pool.load_all()
    pool.warmup_cuda()
    pool._status = "ready"
    print(f"  pool ready in {time.time() - t0:.1f}s", flush=True)

    results = []  # list of dicts: {video, K, trials: [t1,...], median_t, tracks, frames}

    for video_rel in PROBE_VIDEOS:
        vp = VLM4D / video_rel
        if not vp.is_file():
            print(f"  MISSING: {vp}", flush=True)
            continue
        print(f"\n=== Video: {vp.name} ===", flush=True)

        for K in K_VALUES:
            # Mutate the cap.  pool reads cfg at every inference call.
            pool.config.sam3.max_active_tracks = K
            # warmup runs first to absorb compile of any fresh shape
            for _ in range(N_TRIALS_WARMUP):
                pool.run_inference(InferenceRequest(video_path=str(vp), question=None))

            # measured runs
            ts = []
            n_tracks_arr = []
            n_frames = 0
            for _ in range(N_TRIALS_MEASURE):
                t = time.time()
                resp = pool.run_inference(InferenceRequest(video_path=str(vp), question=None))
                dt = time.time() - t
                if resp.status != "ok":
                    print(f"  K={K}: FAILED {resp.error_message}", flush=True)
                    break
                ts.append(dt)
                md = (resp.four_dsg_dict or {}).get("metadata", {})
                n_frames = md.get("num_frames", 0)
                n_tracks_arr.append(md.get("num_tracks", 0))
            if not ts:
                continue
            med_t = statistics.median(ts)
            med_n_tracks = int(statistics.median(n_tracks_arr))
            hz = n_frames / med_t if med_t > 0 else 0
            row = dict(video=vp.name, K=K, trials=ts,
                       median_t=med_t, tracks=med_n_tracks,
                       frames=n_frames, hz=hz)
            results.append(row)
            print(f"  K={K:>2}: tracks={med_n_tracks:>2}  "
                  f"median={med_t:5.2f}s  Hz={hz:5.2f}", flush=True)

    # Save raw data
    out_json = OUT_DIR / "k_scaling.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_json}", flush=True)

    # Plot Hz vs K (one curve per video, plus mean)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)

        # Group by video
        by_vid = {}
        for r in results:
            by_vid.setdefault(r["video"], []).append(r)
        for v in by_vid.values():
            v.sort(key=lambda r: r["K"])

        # Left axis: time vs K
        for vid, rows in by_vid.items():
            Ks = [r["K"] for r in rows]
            ts = [r["median_t"] for r in rows]
            axes[0].plot(Ks, ts, marker="o", label=vid.replace(".mp4", ""))
        axes[0].set_xlabel("max_active_tracks  $K$")
        axes[0].set_ylabel("Per-video time (s)")
        axes[0].set_title("Construction time vs $K$")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=7)

        # Right axis: Hz vs K
        for vid, rows in by_vid.items():
            Ks = [r["K"] for r in rows]
            hz = [r["hz"] for r in rows]
            axes[1].plot(Ks, hz, marker="o", label=vid.replace(".mp4", ""))
        # Average curve
        K_set = sorted({r["K"] for r in results})
        avg_hz = []
        for K in K_set:
            vs = [r["hz"] for r in results if r["K"] == K]
            avg_hz.append(np.mean(vs) if vs else 0)
        axes[1].plot(K_set, avg_hz, marker="s", linestyle="--",
                     color="black", label="mean", linewidth=2)
        axes[1].set_xlabel("max_active_tracks  $K$")
        axes[1].set_ylabel("Effective Hz")
        axes[1].set_title("Throughput vs $K$")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7)

        out_pdf = OUT_DIR / "k_scaling.pdf"
        fig.savefig(str(out_pdf), bbox_inches="tight")
        out_png = OUT_DIR / "k_scaling.png"
        fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
        print(f"Saved {out_pdf}, {out_png}", flush=True)
    except Exception as e:
        print(f"Plot failed (non-fatal): {e}", flush=True)


if __name__ == "__main__":
    main()
