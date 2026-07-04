#!/usr/bin/env python3
"""Test the warm server by running youtube000-009.

Usage:
    python scripts/test_warm_server.py [--compile] [--videos N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_warm")

VIDEO_DIR = ROOT / "benchmark" / "VLM4D-video" / "videos_real" / "youtube-vos"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--videos", type=int, default=10, help="Number of videos (0-9)")
    args = parser.parse_args()

    from rose.engine.config.rose_config import (
        DA3Config, FastSAMConfig, ROSEConfig, SAM3Config, SamplingConfig,
    )
    from rose.engine.server.warm_server import WarmModelPool

    # Config matching diag script
    config = ROSEConfig(
        sampling=SamplingConfig(target_fps=10.0),
        da3=DA3Config(chunk_size=0, chunk_overlap=5),
        fastsam=FastSAMConfig(conf_threshold=0.55, iou_threshold=0.9),
        sam3=SAM3Config(
            offload_state_to_cpu=False,
            offload_video_to_cpu=False,
            max_discovery_total=5,
            max_discovery_per_frame=3,
            chunk_size=-1,
            vg_stride=0,
            enable_compile=args.compile,
        ),
    )

    # ── Load models once ──
    logger.info("=" * 60)
    logger.info("Creating WarmModelPool (compile=%s)...", args.compile)
    t_load_start = time.time()

    pool = WarmModelPool(config)
    pool.load_all()
    t_load = time.time() - t_load_start
    logger.info("Models loaded in %.1fs", t_load)

    # ── CUDA kernel warmup (DA3 + FastSAM) ──
    t_cuda_warmup = pool.warmup_cuda()

    # ── Optional: torch.compile warmup ──
    t_warmup = 0.0
    if args.compile:
        t_warmup = pool.warmup_compile()

    pool._status = "ready"
    logger.info("=" * 60)
    logger.info("Server ready. Starting video tests...")
    logger.info("=" * 60)

    # ── Run youtube videos ──
    from rose.engine.server.warm_server import InferenceRequest

    results = []
    n_videos = min(args.videos, 10)

    for i in range(n_videos):
        video_path = VIDEO_DIR / f"youtube{i:03d}.mp4"
        if not video_path.exists():
            logger.warning("Video not found: %s", video_path)
            continue

        logger.info("")
        logger.info("─" * 50)
        logger.info("Video %d/10: %s", i + 1, video_path.name)
        logger.info("─" * 50)

        req = InferenceRequest(video_path=str(video_path))
        t_start = time.time()
        resp = pool.run_inference(req)
        t_elapsed = time.time() - t_start

        # Extract summary
        n_tracks = 0
        n_frames = 0
        if resp.four_dsg_dict:
            meta = resp.four_dsg_dict.get("metadata", {})
            n_tracks = meta.get("num_tracks", 0)
            n_frames = meta.get("num_frames", 0)

        fps = n_frames / t_elapsed if t_elapsed > 0 else 0
        results.append({
            "video": video_path.name,
            "status": resp.status,
            "time_s": round(t_elapsed, 2),
            "frames": n_frames,
            "tracks": n_tracks,
            "fps": round(fps, 2),
            "error": resp.error_message,
        })

        if resp.status == "ok":
            logger.info(
                "  OK: %d frames, %d tracks, %.2fs (%.2f fps)",
                n_frames, n_tracks, t_elapsed, fps,
            )
        else:
            logger.error("  FAILED: %s", resp.error_message)

        # Clean up keyframe dir
        if resp.keyframe_dir:
            import shutil
            shutil.rmtree(resp.keyframe_dir, ignore_errors=True)

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("  Model load:    %.1fs", t_load)
    if args.compile:
        logger.info("  Compile warmup: %.1fs", t_warmup)
    logger.info("")
    logger.info("  %-15s %8s %6s %6s %8s", "Video", "Time(s)", "Frames", "Tracks", "FPS")
    logger.info("  " + "-" * 50)

    total_time = 0
    total_frames = 0
    ok_count = 0
    for r in results:
        logger.info(
            "  %-15s %8.2f %6d %6d %8.2f  %s",
            r["video"], r["time_s"], r["frames"], r["tracks"], r["fps"],
            "" if r["status"] == "ok" else f"ERROR: {r['error']}",
        )
        if r["status"] == "ok":
            total_time += r["time_s"]
            total_frames += r["frames"]
            ok_count += 1

    logger.info("  " + "-" * 50)
    avg_fps = total_frames / total_time if total_time > 0 else 0
    logger.info("  %-15s %8.2f %6d %6s %8.2f", "TOTAL", total_time, total_frames, "", avg_fps)
    logger.info("")
    logger.info("  OK: %d/%d videos", ok_count, len(results))
    logger.info("  Average FPS: %.2f", avg_fps)
    logger.info("=" * 60)

    # Cleanup
    pool.unload_all()


if __name__ == "__main__":
    main()
