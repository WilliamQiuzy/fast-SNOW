"""Benchmark per-component latency for streaming pipeline design.

Measures real wall-clock time for each module on V100-32GB using horsing.mp4.
Results are used to validate the streaming pipeline timing in DA3_BATCH_OOM.md.

Usage:
    conda activate snow
    PYTHONPATH=. python scripts/benchmark_latency.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO_PATH = ROOT / "assets" / "examples_videos" / "horsing.mp4"
DA3_MODEL = ROOT / "fast_snow" / "models" / "da3-small"
SAM3_MODEL = ROOT / "fast_snow" / "models" / "sam3"
YOLO_MODEL = "yolo11n.pt"


def extract_frames(video_path: Path, fps: float, max_frames: int) -> list:
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(src_fps / fps)))
    frames = []
    idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while idx < total and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        idx += step
    cap.release()
    return frames


def save_frames_as_jpeg(frames: list, out_dir: Path) -> Path:
    """Save frames as numbered JPEGs (required by SAM3 video dir)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    return out_dir


def benchmark_da3(frames: list):
    """Benchmark DA3 batch inference at various chunk sizes."""
    import torch
    from fast_snow.engine.config.fast_snow_config import DA3Config
    from fast_snow.vision.perception.da3_wrapper import DA3Wrapper

    print("\n" + "=" * 60)
    print("DA3 BENCHMARK")
    print("=" * 60)

    # Warmup
    cfg = DA3Config(chunk_size=0)
    wrapper = DA3Wrapper(cfg)
    wrapper.load()
    _ = wrapper.infer(frames[0])
    torch.cuda.synchronize()

    n = len(frames)
    chunk_sizes = [5, 8, 10, 15, 20]

    for cs in chunk_sizes:
        test_frames = frames[:cs]
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = wrapper._infer_batch_core(test_frames)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        print(f"  chunk_size={cs:3d}: {elapsed:.3f}s total, {elapsed/cs*1000:.1f}ms/frame")

    # Also test chunked inference
    print()
    for total_frames, cs, overlap in [(20, 10, 3), (30, 10, 3), (30, 15, 5)]:
        test_frames = frames[:total_frames]
        cfg_ch = DA3Config(chunk_size=cs, chunk_overlap=overlap)
        wrapper_ch = DA3Wrapper(cfg_ch)
        wrapper_ch._model = wrapper._model  # reuse loaded model

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = wrapper_ch.infer_batch(test_frames)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        print(f"  chunked {total_frames} frames (chunk={cs}, overlap={overlap}): "
              f"{elapsed:.3f}s total, {elapsed/total_frames*1000:.1f}ms/frame")


def benchmark_yolo(frames: list):
    """Benchmark YOLO per-frame inference."""
    import torch
    from fast_snow.vision.perception.yolo_wrapper import YoloBBoxDetector

    print("\n" + "=" * 60)
    print("YOLO BENCHMARK")
    print("=" * 60)

    detector = YoloBBoxDetector()
    detector.load()

    # Warmup (3 frames)
    for i in range(min(3, len(frames))):
        detector.detect(frames[i], frame_idx=i)
    torch.cuda.synchronize()

    # Benchmark
    n = min(30, len(frames))
    times = []
    det_counts = []
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dets = detector.detect(frames[i], frame_idx=i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        det_counts.append(len(dets))

    times_ms = [t * 1000 for t in times]
    print(f"  Frames tested: {n}")
    print(f"  Mean: {np.mean(times_ms):.1f}ms  Median: {np.median(times_ms):.1f}ms  "
          f"Min: {np.min(times_ms):.1f}ms  Max: {np.max(times_ms):.1f}ms")
    print(f"  Detections per frame: mean={np.mean(det_counts):.1f}, "
          f"min={np.min(det_counts)}, max={np.max(det_counts)}")


def benchmark_sam3(frames: list, frame_dir: Path):
    """Benchmark SAM3 init + propagation."""
    import torch
    from fast_snow.vision.perception.yolo_wrapper import YoloBBoxDetector
    from fast_snow.vision.perception.sam3_shared_session_wrapper import SAM3SharedSessionManager

    print("\n" + "=" * 60)
    print("SAM3 BENCHMARK")
    print("=" * 60)

    # First get YOLO bboxes for frame 0
    yolo = YoloBBoxDetector()
    yolo.load()
    dets = yolo.detect(frames[0], frame_idx=0)
    bboxes = [list(d.bbox_xywh_norm) for d in dets]
    print(f"  YOLO detected {len(bboxes)} objects in frame 0")

    if not bboxes:
        print("  SKIP: no YOLO detections for SAM3 init")
        return

    # Load SAM3
    sam3 = SAM3SharedSessionManager()
    t_load_start = time.perf_counter()
    sam3.load()
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t_load_start
    print(f"  SAM3 model load: {t_load:.3f}s")

    # Set video dir
    sam3.set_video_dir(frame_dir)

    # Init with bboxes
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, init_masks = sam3.create_run_with_initial_bboxes(
        boxes_xywh=bboxes,
        box_labels=[1] * len(bboxes),
        frame_idx=0,
        tag="benchmark",
    )
    torch.cuda.synchronize()
    t_init = time.perf_counter() - t0
    print(f"  SAM3 init (add_prompt, {len(bboxes)} bboxes): {t_init:.3f}s → {len(init_masks)} masks")

    # Propagation: this is the critical measurement.
    # SAM3's propagate_all does a single propagate_in_video for ALL remaining frames.
    # We measure the total time AND per-frame yield time.
    n = len(frames)
    print(f"  Propagating through {n-1} frames...")

    # propagate_all(frame_idx=1) will trigger propagate_in_video for ALL frames
    torch.cuda.synchronize()
    t_prop_start = time.perf_counter()
    masks_f1 = sam3.propagate_all(frame_idx=1)
    torch.cuda.synchronize()
    t_prop_total = time.perf_counter() - t_prop_start
    print(f"  SAM3 propagate_all(frame 1): {t_prop_total:.3f}s "
          f"(propagates ALL {n-1} remaining frames, caches results)")
    print(f"  Frame 1 masks: {len(masks_f1)}")

    # Subsequent frames are just cache lookups
    cache_times = []
    for fi in range(2, n):
        t0 = time.perf_counter()
        masks = sam3.propagate_all(frame_idx=fi)
        elapsed = time.perf_counter() - t0
        cache_times.append(elapsed * 1000)

    if cache_times:
        print(f"  Cache lookup (frame 2-{n-1}): "
              f"mean={np.mean(cache_times):.3f}ms, max={np.max(cache_times):.3f}ms")

    # Per-frame propagation time = total / (n-1)
    per_frame_ms = t_prop_total / (n - 1) * 1000
    print(f"  Effective per-frame propagation: {per_frame_ms:.1f}ms/frame "
          f"(= {t_prop_total:.3f}s / {n-1} frames)")

    sam3.end_all_runs()


def benchmark_cpu_pipeline(frames: list):
    """Benchmark CPU-side pipeline (backproject + STEP + fusion)."""
    from fast_snow.engine.config.fast_snow_config import FastSNOWConfig, DA3Config
    from fast_snow.engine.pipeline.fast_snow_pipeline import (
        FastSNOWPipeline, FastFrameInput, FastLocalDetection,
    )
    from fast_snow.vision.perception.da3_wrapper import DA3Wrapper

    print("\n" + "=" * 60)
    print("CPU PIPELINE BENCHMARK (backproject + STEP + fusion)")
    print("=" * 60)

    # Get DA3 results first
    import torch
    cfg = DA3Config(chunk_size=0)
    da3 = DA3Wrapper(cfg)
    da3.load()
    n = min(15, len(frames))
    test_frames = frames[:n]
    da3_results = da3._infer_batch_core(test_frames)
    torch.cuda.synchronize()

    # Create synthetic masks (since we're benchmarking CPU pipeline, not SAM3)
    H, W = frames[0].shape[:2]
    config = FastSNOWConfig()
    pipeline = FastSNOWPipeline(config)

    times = []
    for i in range(n):
        # Simulate 5 object masks per frame
        dets = []
        for obj_id in range(5):
            mask = np.zeros((H, W), dtype=bool)
            y0, x0 = np.random.randint(0, H - 100), np.random.randint(0, W - 100)
            mask[y0:y0 + 80, x0:x0 + 80] = True
            dets.append(FastLocalDetection(
                run_id=0, local_obj_id=obj_id, mask=mask, score=0.9,
            ))

        fi = FastFrameInput(
            frame_idx=i,
            depth_t=da3_results[i].depth,
            K_t=da3_results[i].K,
            T_wc_t=da3_results[i].T_wc,
            detections=dets,
            depth_conf_t=da3_results[i].depth_conf,
            depth_is_metric=da3_results[i].is_metric,
        )

        t0 = time.perf_counter()
        pipeline.process_frame(fi)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times_ms = [t * 1000 for t in times]
    print(f"  Frames: {n}, Objects/frame: 5")
    print(f"  Mean: {np.mean(times_ms):.1f}ms  Median: {np.median(times_ms):.1f}ms  "
          f"Min: {np.min(times_ms):.1f}ms  Max: {np.max(times_ms):.1f}ms")

    # Breakdown: just backproject
    from fast_snow.reasoning.tokens.patch_tokenizer import mask_to_patch_tokens
    from fast_snow.reasoning.tokens.geometry_tokens import build_centroid_token, build_shape_token

    mask = np.zeros((H, W), dtype=bool)
    mask[100:300, 200:500] = True
    depth = da3_results[0].depth
    K = da3_results[0].K
    T_cw = np.linalg.inv(da3_results[0].T_wc)

    # Backproject timing
    bp_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        pipeline._backproject_mask_points(mask, depth, K, T_cw, da3_results[0].depth_conf)
        bp_times.append((time.perf_counter() - t0) * 1000)
    print(f"  Backproject single mask: mean={np.mean(bp_times):.2f}ms")

    # STEP token timing
    step_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        mask_to_patch_tokens(mask, grid_size=16, iou_threshold=0.5)
        step_times.append((time.perf_counter() - t0) * 1000)
    print(f"  STEP patch tokenize: mean={np.mean(step_times):.2f}ms")


def main():
    print("Extracting frames from horsing.mp4...")
    frames = extract_frames(VIDEO_PATH, fps=10.0, max_frames=30)
    print(f"Extracted {len(frames)} frames, resolution {frames[0].shape[1]}x{frames[0].shape[0]}")

    import tempfile
    frame_dir = Path(tempfile.mkdtemp(prefix="bench_sam3_"))
    save_frames_as_jpeg(frames, frame_dir)
    print(f"Saved JPEGs to {frame_dir}")

    try:
        benchmark_da3(frames)
        benchmark_yolo(frames)
        benchmark_sam3(frames, frame_dir)
        benchmark_cpu_pipeline(frames)
    finally:
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
