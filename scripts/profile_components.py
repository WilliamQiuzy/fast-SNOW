"""Profile each pipeline component for the two benchmark videos using the
currently-shipped config. Captures:
  - Video duration / total frames
  - Sampled frame count
  - Per-phase wall-clock time (B-1 FastSAM, B-2 dedup, B-3 init, B-4 VG
    propagate, B-5 refine, B-6 memory propagate, B-7.5 re-prompt, dedup,
    4DSG build, DA3 depth)
  - Total time per video
"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time, re
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import cv2
import torch
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"path": path, "fps": fps, "n_frames": n, "duration_s": n/fps if fps else 0, "w": w, "h": h}


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    # SHIPPED defaults from rose_config.py:
    #   compile_memory_encoder = True
    #   compile_mask_decoder_transformer = True
    #   sampling.max_frames = 9, etc.
    print("Loading pool (shipped defaults)...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"

    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]
    print("\n=== Video metadata ===")
    for v in videos:
        info = video_info(v)
        print(f"  {Path(v).name}: {info['w']}x{info['h']}, "
              f"{info['n_frames']} frames @ {info['fps']:.1f}fps = {info['duration_s']:.2f}s")

    print(f"\nSampled frames per video: {cfg.sampling.max_frames}")
    print(f"  (target_fps={cfg.sampling.target_fps} → 9 keyframes uniformly sampled)")

    # Intercept stderr to capture [TIMING] markers per video. Phase logs
    # emitted via logger.info also have "Phase B-..." prefixes — capture both.
    import io
    class TeeStderr:
        def __init__(self, real):
            self.real = real
            self.buf = io.StringIO()
        def write(self, s):
            self.buf.write(s); return self.real.write(s)
        def flush(self): self.real.flush()
        def isatty(self): return False

    print("\n=== Warmup ==="); pool.run_inference(InferenceRequest(video_path=videos[0], question=None))

    for v in videos:
        print(f"\n=== Profiling {Path(v).name} ===")
        sys.stderr.flush()
        tee = TeeStderr(sys.stderr); sys.stderr = tee
        # Also intercept the rose logger output by attaching a logging handler
        import logging
        log_buf = io.StringIO()
        h = logging.StreamHandler(log_buf)
        h.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(h)
        t0 = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
        wall = time.time() - t0
        logging.getLogger().removeHandler(h)
        sys.stderr = tee.real
        captured = tee.buf.getvalue() + log_buf.getvalue()

        # Parse [TIMING] markers + "Phase B-X:" logger lines
        phase_times = defaultdict(float)
        for line in captured.splitlines():
            m = re.search(r"\[TIMING\] Phase ([^:]+):\s*([\d.]+)s", line)
            if m:
                phase = re.sub(r"\s*\(.*$", "", m.group(1)).strip()
                phase = re.sub(r"\s+\d+\s+(objects?|matched|FA3|init).*", "", phase)
                phase_times[phase] += float(m.group(2))
                continue
            m = re.search(r"Phase (B-\d+(\.\d+)?)(?:[^:]*):.*in ([\d.]+)s", line)
            if m:
                phase_times[m.group(1)] += float(m.group(3))

        print(f"  Wall time: {wall:.2f}s")
        print("  Component breakdown:")
        accounted = 0
        for k, t in sorted(phase_times.items(), key=lambda x: -x[1]):
            print(f"    {k:<35} {t:>6.3f}s  ({100*t/wall:5.1f}%)")
            accounted += t
        other = wall - accounted
        print(f"    {'(other: DA3 depth + dedup + 4DSG build + IO)':<35} {other:>6.3f}s  ({100*other/wall:5.1f}%)")
        sg = resp.four_dsg_dict
        if sg:
            n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True)
            print(f"  Quality: {len(n)} tracks, lengths = {n}")


if __name__ == "__main__":
    main()
