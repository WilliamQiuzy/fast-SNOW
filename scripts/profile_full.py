"""Fine-grained per-component profiler.

Reports for each video:
  1. DA3 monocular depth
  2. FastSAM anchor detection
  3. SAM 3.1 components (B-1 ... B-7.5 + breakdown)
  4. 4DSG build (mask preprocessing + scene-graph fusion)
  5. 3D lifting (mask + depth → world points, inside 4DSG)
  6. VLM inference

Sampling: 10 fps × video duration (no max_frames cap).
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import cv2
import torch
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


# Timings registered by hooks. Cleared per video.
T = defaultdict(float)


def timeit(name):
    """Decorator/context manager that adds CUDA-sync timing to `T[name]`."""
    class _Ctx:
        def __enter__(self):
            torch.cuda.synchronize()
            self.t0 = time.time()
        def __exit__(self, *a):
            torch.cuda.synchronize()
            T[name] += time.time() - self.t0
    return _Ctx()


def install_hooks(pool):
    """Monkey-patch the entry point of each component to record wall time."""

    # ── DA3 depth ──
    orig_da3 = pool._da3.infer_batch_chunked
    def da3_timed(*args, **kwargs):
        with timeit("DA3 depth"):
            return orig_da3(*args, **kwargs)
    pool._da3.infer_batch_chunked = da3_timed

    # ── FastSAM ──
    if hasattr(pool._fastsam, "detect_batch"):
        orig_fs = pool._fastsam.detect_batch
        def fs_timed(*args, **kwargs):
            with timeit("FastSAM (anchors)"):
                return orig_fs(*args, **kwargs)
        pool._fastsam.detect_batch = fs_timed

    # ── SAM3 sub-phases: instrument explicit call points ──
    # The pipeline already prints [TIMING] for B-4/B-5/B-6/B-7.5 via _t().
    # We re-implement by hooking the wrapper methods.
    sam3 = pool._sam3
    for method, label in [
        ("add_bboxes_batch", "SAM3 B-3 init add_bboxes_batch"),
        ("propagate_new_objects", "SAM3 propagate_new_objects"),
        ("propagate_all", "SAM3 propagate_all"),
        ("refine_object_with_point", "SAM3 B-5 refine point"),
        ("add_object_point", "SAM3 add_object_point"),
    ]:
        if hasattr(sam3, method):
            orig = getattr(sam3, method)
            label_ = label
            def make(orig_f, lab):
                def hook(*args, **kwargs):
                    with timeit(lab):
                        return orig_f(*args, **kwargs)
                return hook
            setattr(sam3, method, make(orig, label_))

    # ── 4DSG build ──
    orig_b4d = pool._build_4dsg
    def b4d_timed(*args, **kwargs):
        with timeit("4DSG build (total)"):
            return orig_b4d(*args, **kwargs)
    pool._build_4dsg = b4d_timed

    # ── 3D lifting: backproject_mask_points inside ROSEPipeline ──
    from rose.engine.pipeline.rose_pipeline import ROSEPipeline
    if not hasattr(ROSEPipeline, "_backproject_mask_points_orig"):
        ROSEPipeline._backproject_mask_points_orig = ROSEPipeline._backproject_mask_points
        def bp_timed(self, *args, **kwargs):
            with timeit("3D lifting (backproject)"):
                return ROSEPipeline._backproject_mask_points_orig(self, *args, **kwargs)
        ROSEPipeline._backproject_mask_points = bp_timed

    # ── VLM ──
    orig_vlm = pool._query_vlm
    def vlm_timed(*args, **kwargs):
        with timeit("VLM inference"):
            return orig_vlm(*args, **kwargs)
    pool._query_vlm = vlm_timed


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n / fps if fps else 0, n, fps


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    # All compile+FA3 patches already in shipped defaults.
    # max_frames=None now (set in rose_config.py)

    print("Loading pool (max_frames=None, target_fps=10, compiles+FA3 patches)...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    install_hooks(pool)

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]
    # VLM needs API key — leave None unless env var is set
    question = None
    if os.environ.get(cfg.vlm.api_key_env if hasattr(cfg, "vlm") else "VLM_API_KEY"):
        question = "What is happening in this video?"

    # Warmup (triggers torch.compile etc., not timed)
    print("Warmup..."); pool.run_inference(InferenceRequest(video_path=videos[0], question=question))
    T.clear()

    print("\n" + "=" * 90)
    for v in videos:
        dur, n_total, fps = video_info(v)
        expected = int(round(dur * 10))  # target_fps=10
        T.clear()
        torch.cuda.synchronize(); t0 = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=v, question=question))
        torch.cuda.synchronize(); wall = time.time() - t0
        sg = resp.four_dsg_dict
        n_tracks = len(sg["tracks"]) if sg else 0
        track_lens = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
        sampled = max(track_lens) if track_lens else expected

        print(f"\n### {Path(v).name}")
        print(f"  Video duration:  {dur:.2f}s ({n_total} frames @ {fps:.1f}fps)")
        print(f"  Sampled frames:  {sampled} (target_fps=10 → expected ~{expected})")
        print(f"  Wall time:       {wall:.3f}s ({1/wall:.3f} Hz)")
        print(f"  Quality:         {n_tracks} tracks, lengths={track_lens}")
        print(f"  Component times:")
        # Top-level (independent) buckets. "3D lifting" is INSIDE "4DSG build",
        # so don't add 3D lifting to the accounted total — show it as a
        # sub-line under 4DSG.
        toplevel_order = [
            ("FastSAM (anchors)",            "FastSAM (anchors)"),
            ("DA3 depth",                    "DA3 depth"),
            ("SAM3 B-3 init add_bboxes_batch", "SAM3 B-3 init (add_bboxes_batch)"),
            ("SAM3 propagate_new_objects",   "SAM3 propagate (B-4 grounding + B-6 memory + B-7.5 re-prompt)"),
            ("SAM3 propagate_all",           "SAM3 propagate_all"),
            ("SAM3 B-5 refine point",        "SAM3 B-5 refine"),
            ("SAM3 add_object_point",        "SAM3 add_object_point (B-8 late discovery)"),
            ("4DSG build (total)",           "4DSG build (incl. 3D lifting below)"),
            ("VLM inference",                "VLM inference"),
        ]
        accounted = 0.0
        for key, label in toplevel_order:
            t = T.get(key, 0.0)
            if t == 0.0:
                if key == "VLM inference":
                    print(f"    {label:<60} skipped (no API key)")
                continue
            print(f"    {label:<60} {t:>7.3f}s  ({100*t/wall:5.1f}%)")
            accounted += t
        # 3D lifting is a sub-component of 4DSG build — show indented
        t_lift = T.get("3D lifting (backproject)", 0.0)
        if t_lift > 0:
            print(f"      └─ 3D lifting (backproject)                         "
                  f"{t_lift:>7.3f}s  ({100*t_lift/wall:5.1f}% of total)")
        # other components (not hooked)
        other = wall - accounted
        print(f"    {'(other: preprocess, dedup, mask post-proc, IO)':<60} {other:>7.3f}s  ({100*other/wall:5.1f}%)")


if __name__ == "__main__":
    main()
