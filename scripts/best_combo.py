"""Test best combo from sweep: max_init=20 + mem_stride=4"""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"rose/vision/sam3"))
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest

cfg = ROSEConfig()
cfg.da3.model_path = "rose/models/da3-small"
cfg.sam3.use_fa3 = True
cfg.sam3.use_multiplex = True
cfg.sam3.enable_compile = False
cfg.sampling.max_frames = 32
# Best combo
cfg.sam3.max_init_masks = 20
cfg.sam3.memory_temporal_stride = 4

print("Loading...")
pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
videos = [
    "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
    "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    "/workspace/fast-SNOW/sample_videos_and_analysis/PartLevel-3-WheelState.mp4",
    "/workspace/fast-SNOW/sample_videos_and_analysis/PartLevel-4-ArmRotation.mp4",
]
print("Warm-up...")
pool.run_inference(InferenceRequest(video_path=videos[0], question=None))

print(f"\n{'video':<28} {'time':<8} {'tracks':<8} {'long':<6} {'max':<5}")
print("-" * 60)
all_dt = []
for v in videos:
    sg = None
    for _ in range(2):  # 2 runs each, take min
        t = time.time()
        resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
        dt = time.time() - t
        sg = resp.four_dsg_dict if resp.status == "ok" else None
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    print(f"{Path(v).stem:<28} {dt:<8.2f} {len(n):<8} "
          f"{sum(1 for x in n if x>=20):<6} {max(n, default=0):<5}")
    all_dt.append(dt)

print(f"\nMean time: {sum(all_dt)/len(all_dt):.2f}s")
print(f"Hz at mean: {1.0/(sum(all_dt)/len(all_dt)):.4f}")
print(f"Min time: {min(all_dt):.2f}s = {1.0/min(all_dt):.4f} Hz")
