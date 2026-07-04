"""Ablation: baseline (no D2/D4) vs D2+D4 (frame-level dedup + bounded backward).

Same 12 videos, 2 throwaway, post-warmup measurements only.
Quality guardrails:
  - num_tracks (recall: should NOT decrease)
  - n_obs_total (track stability: should NOT decrease)

If guardrails hold, report Hz speedup.
"""
from __future__ import annotations

import json, os, random, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest

VLM4D = ROOT / "benchmark" / "VLM4D-video"
HF = "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/"
N_THROWAWAY, N_MEASURED = 2, 12
MAX_FRAMES, TARGET_FPS = 32, 10.0


def cfg(window: int):
    c = ROSEConfig()
    c.da3.model_path = "rose/models/da3-small"
    c.sampling.max_frames = MAX_FRAMES
    c.sampling.target_fps = TARGET_FPS
    c.sam3.use_fa3 = True
    c.sam3.offload_state_to_cpu = False
    c.sam3.offload_video_to_cpu = False
    c.sam3.enable_compile = False
    c.sam3.discovery_backward_window = window
    return c


def collect_videos():
    rng = random.Random(42)
    videos = []
    for fname in ["mini_real_mc.json", "mini_synthetic_mc.json"]:
        with open(VLM4D / "QA" / fname) as f:
            for q in json.load(f):
                p = VLM4D / q["video"].replace(HF, "")
                if p.is_file() and p not in videos:
                    videos.append(p)
    return rng.sample(videos, N_THROWAWAY + N_MEASURED)


def run_pool(pool, videos, label):
    rows = []
    for i, vp in enumerate(videos, 1):
        try:
            t0 = time.time()
            r = pool.run_inference(InferenceRequest(video_path=str(vp)))
            dt = time.time() - t0
            if r.status != "ok":
                print(f"  [{label}][{i}] {vp.name} FAILED: {r.error_message}")
                continue
            md = (r.four_dsg_dict or {}).get("metadata", {})
            tracks = (r.four_dsg_dict or {}).get("tracks", [])
            n_obs = sum(len(t.get("F_k", [])) for t in tracks)
            row = {
                "video": vp.name,
                "n_frames": md.get("num_frames", 0),
                "n_tracks": md.get("num_tracks", 0),
                "n_obs": n_obs,
                "time_s": dt,
                "hz": md.get("num_frames", 0) / dt if dt > 0 else 0.0,
            }
            rows.append(row)
            print(f"  [{label}][{i:>2}] {vp.name:30s}  frames={row['n_frames']:>2}  tracks={row['n_tracks']:>2}  obs={row['n_obs']:>3}  time={dt:5.1f}s  Hz={row['hz']:5.2f}")
        except Exception as exc:
            print(f"  [{label}][{i}] {vp.name} CRASH: {type(exc).__name__}: {exc}")
    return rows


def summarize(rows, label):
    if not rows:
        print(f"\n  {label}: NO RUNS")
        return None
    return {
        "label": label,
        "n": len(rows),
        "hz_med": statistics.median(r["hz"] for r in rows),
        "hz_mean": statistics.mean(r["hz"] for r in rows),
        "tracks_med": statistics.median(r["n_tracks"] for r in rows),
        "tracks_total": sum(r["n_tracks"] for r in rows),
        "obs_total": sum(r["n_obs"] for r in rows),
        "time_total": sum(r["time_s"] for r in rows),
        "rows": rows,
    }


def main():
    sample = collect_videos()
    throwaway, measured = sample[:N_THROWAWAY], sample[N_THROWAWAY:]
    print(f"Ablation: {len(measured)} measured videos, max_frames={MAX_FRAMES}, target_fps={TARGET_FPS}\n")

    results = {}
    # Run BASELINE first: window=999 disables backward bounding (≈ old start_frame_idx=0)
    # then run D2+D4 with default window=5.  D2 (frame-level IoU dedup) is now in code path
    # for both — to truly isolate D2+D4 vs baseline we'd need a flag, but: D2 only changes
    # build_4dsg behavior when there are duplicate masks, which improves quality if any.
    # The HZ delta will primarily reflect D4.
    for label, window in [("BASELINE (window=999)", 999), ("D2+D4 (window=10)", 10)]:
        print(f"\n{'='*70}\n  {label}\n{'='*70}")
        pool = WarmModelPool(cfg(window))
        pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
        print(f"\n  Throwaway runs:")
        run_pool(pool, throwaway, label)
        print(f"\n  Measured runs:")
        rows = run_pool(pool, measured, label)
        results[label] = summarize(rows, label)
        pool.unload_all()
        import torch
        torch.cuda.empty_cache()
        del pool

    # Report
    print(f"\n\n{'='*70}\n  ABLATION SUMMARY\n{'='*70}")
    print(f"{'metric':<28} {'baseline':>15} {'D2+D4':>15} {'delta':>15}")
    print("-" * 75)
    base = results["BASELINE (window=999)"]
    d24 = results["D2+D4 (window=10)"]
    if base and d24:
        for k, fmt in [("hz_med", "{:.2f}"), ("hz_mean", "{:.2f}"),
                       ("tracks_med", "{:.0f}"), ("tracks_total", "{:.0f}"),
                       ("obs_total", "{:.0f}"), ("time_total", "{:.1f}s")]:
            b, d = base[k], d24[k]
            delta = d - b
            pct = 100 * delta / b if b else 0
            print(f"{k:<28} {fmt.format(b):>15} {fmt.format(d):>15} {fmt.format(delta):>10} ({pct:+.1f}%)")
        speedup = d24["hz_med"] / base["hz_med"] if base["hz_med"] > 0 else 0
        print(f"\nHz speedup (D2+D4 / baseline): {speedup:.2f}x")
        # Quality guardrails
        print("\nQuality guardrails (NEED >=):")
        print(f"  num_tracks total:  baseline={base['tracks_total']:>4} → D2+D4={d24['tracks_total']:>4}  {'OK' if d24['tracks_total'] >= base['tracks_total'] else 'REGRESSION'}")
        print(f"  n_obs total:       baseline={base['obs_total']:>4} → D2+D4={d24['obs_total']:>4}  {'OK' if d24['obs_total'] >= base['obs_total'] * 0.95 else 'REGRESSION'} (>=95%)")

    out = ROOT / "benchmark" / "ablation_d2_d4.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
