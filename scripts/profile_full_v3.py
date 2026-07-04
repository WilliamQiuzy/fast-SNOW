"""Like profile_full.py but splits SAM3 propagate into B-1/B-2/B-3/B-4/B-5/
B-6/B-7.5/B-8 by capturing the existing [TIMING] markers warm_server emits
to stderr.
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time, io, re
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import cv2
import torch
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


T = defaultdict(float)


def timeit(name):
    class _Ctx:
        def __enter__(self):
            torch.cuda.synchronize(); self.t0 = time.time()
        def __exit__(self, *a):
            torch.cuda.synchronize(); T[name] += time.time() - self.t0
    return _Ctx()


def install_hooks(pool):
    # DA3
    orig = pool._da3.infer_batch_chunked
    def da3_t(*a, **k):
        with timeit("DA3 depth"): return orig(*a, **k)
    pool._da3.infer_batch_chunked = da3_t

    # FastSAM (already timed by warm_server's "Phase B-1" log)
    if hasattr(pool._fastsam, "detect_batch"):
        o = pool._fastsam.detect_batch
        def fs(*a, **k):
            with timeit("FastSAM anchors"): return o(*a, **k)
        pool._fastsam.detect_batch = fs

    # 4DSG build
    orig_b = pool._build_4dsg
    def b4d(*a, **k):
        with timeit("4DSG build (total)"): return orig_b(*a, **k)
    pool._build_4dsg = b4d

    # 3D lifting (inside 4DSG)
    from rose.engine.pipeline.rose_pipeline import ROSEPipeline
    if not hasattr(ROSEPipeline, "_backproject_mask_points_orig_v3"):
        ROSEPipeline._backproject_mask_points_orig_v3 = ROSEPipeline._backproject_mask_points
        def bp(self, *a, **k):
            with timeit("3D lifting (backproject)"):
                return ROSEPipeline._backproject_mask_points_orig_v3(self, *a, **k)
        ROSEPipeline._backproject_mask_points = bp

    # VLM
    orig_v = pool._query_vlm
    def vlm(*a, **k):
        with timeit("VLM inference"): return orig_v(*a, **k)
    pool._query_vlm = vlm


def parse_timing_markers(stream_text):
    """Aggregate seconds reported via warm_server's '[TIMING] Phase B-...'
    stderr lines.  These cover B-4, B-5, B-6, B-7.5, B-8 sub-phases.
    Also captures the 'Phase B-1: ... in Xs', 'B-2: ... in Xs', etc info logs.
    """
    by_phase = defaultdict(float)
    for line in stream_text.splitlines():
        m = re.search(r"\[TIMING\] Phase ([^:]+):\s*([\d.]+)s", line)
        if m:
            phase = m.group(1)
            phase = re.sub(r"\s*\(.*$", "", phase)
            phase = re.sub(r"\s+\d+\s+(objects?|matched|FA3|init|objs).*", "", phase)
            by_phase[phase.strip()] += float(m.group(2))
            continue
        m2 = re.search(r"Phase (B-\d+(?:\.\d+)?)[^:]*:.*?(?:in\s+)?([\d.]+)s", line)
        if m2:
            ph = m2.group(1)
            by_phase[ph + " (info log)"] += float(m2.group(2))
    return by_phase


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
    print("Loading pool (10fps sampling, quality-safe defaults, compile+FA3 on)...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    install_hooks(pool)

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    print("Warmup..."); pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
    T.clear()

    print("\n" + "=" * 90)
    for v in videos:
        dur, n_total, fps = video_info(v)

        # Capture stderr for the [TIMING] markers
        import io
        old_stderr = sys.stderr
        tee = io.StringIO()
        class Tee:
            def write(self, s):
                tee.write(s); return old_stderr.write(s)
            def flush(self): old_stderr.flush()
            def isatty(self): return False
        sys.stderr = Tee()
        # Also capture rose logger output
        import logging
        log_buf = io.StringIO()
        h = logging.StreamHandler(log_buf); h.setLevel(logging.INFO)
        logging.getLogger().addHandler(h)

        T.clear()
        torch.cuda.synchronize(); t0 = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
        torch.cuda.synchronize(); wall = time.time() - t0

        logging.getLogger().removeHandler(h)
        sys.stderr = old_stderr
        stderr_text = tee.getvalue() + log_buf.getvalue()
        phases = parse_timing_markers(stderr_text)

        sg = resp.four_dsg_dict
        n_tracks = len(sg["tracks"]) if sg else 0
        track_lens = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
        sampled = max(track_lens) if track_lens else 0

        print(f"\n### {Path(v).name}")
        print(f"  Video duration:  {dur:.2f}s ({n_total} frames @ {fps:.1f}fps)")
        print(f"  Sampled frames:  {sampled} (target_fps=10 → expected ~{int(round(dur*10))})")
        print(f"  Wall time:       {wall:.3f}s ({1/wall:.3f} Hz)")
        print(f"  Quality:         {n_tracks} tracks, lengths={track_lens[:8]}{'...' if len(track_lens)>8 else ''}")

        # SAM3 sub-phases from stderr
        sam3_total = sum(phases.values())
        print(f"\n  --- SAM3 sub-phases (from [TIMING] markers) ---")
        for ph, t in sorted(phases.items(), key=lambda x: -x[1]):
            print(f"    {ph:<55} {t:>7.3f}s  ({100*t/wall:5.1f}%)")
        print(f"    {'SAM3 SUB-PHASES TOTAL':<55} {sam3_total:>7.3f}s  ({100*sam3_total/wall:5.1f}%)")

        # Top-level (non-SAM3) buckets — DA3, FastSAM, 4DSG, 3D lift, VLM
        print(f"\n  --- Other components ---")
        accounted_extra = 0.0
        for k, label in [
            ("DA3 depth",                  "DA3 depth"),
            ("FastSAM anchors",            "FastSAM anchors"),
            ("4DSG build (total)",         "4DSG build (incl. 3D lifting below)"),
            ("VLM inference",              "VLM inference"),
        ]:
            t = T.get(k, 0.0)
            if t == 0.0:
                if k == "VLM inference":
                    print(f"    {label:<55} skipped (no API key)")
                continue
            print(f"    {label:<55} {t:>7.3f}s  ({100*t/wall:5.1f}%)")
            accounted_extra += t
        # 3D lift is sub-component of 4DSG
        t_lift = T.get("3D lifting (backproject)", 0.0)
        if t_lift > 0:
            print(f"      └─ 3D lifting (backproject)                          {t_lift:>7.3f}s  ({100*t_lift/wall:5.1f}% of total)")

        # Remainder (preprocess, dedup, mask post-proc, IO, gluecode)
        other = wall - sam3_total - accounted_extra
        print(f"\n  --- Remainder (preprocess + IO + glue) ---")
        print(f"    {'other':<55} {other:>7.3f}s  ({100*other/wall:5.1f}%)")


if __name__ == "__main__":
    main()
