"""Stand-alone VLM-only timing for the rebuttal.

Reuses pre-computed 4DSG JSONs from benchmark/fdsg_*/ (no SAM3/DA3 work).
Loads Qwen2.5-VL-32B once and times paper-spec inference with the FULL
4DSG: every anchor crop, every keyframe (sampled from the source video),
full per-track trajectory, max_new_tokens=512, no constrained decode.

Outputs:
  - benchmark/vlm_only_timing.json
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
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import cv2
import numpy as np
from PIL import Image

QWEN_PATH = ROOT / "rose" / "models" / "qwen2.5-vl-32b-instruct"
VLM4D = ROOT / "benchmark" / "VLM4D-video"
DSR_DIR = ROOT / "benchmark" / "DSR-Bench" / "videos"

# Pair each 4DSG JSON with its source video so we can sample 32 keyframes.
SAMPLES = [
    {
        "name": "DSR (V3o32F957TA_17)",
        "fdsg": ROOT / "benchmark" / "fdsg_compare" / "base_sam3" / "synth_synth_241.4dsg.json",
        "video": VLM4D / "videos_synthetic" / "synth_241.mp4",
    },
    {
        "name": "VLM4D (synth_350)",
        "fdsg": ROOT / "benchmark" / "fdsg_samples_pipelined" / "synth_synth_241.4dsg.json",
        "video": VLM4D / "videos_synthetic" / "synth_241.mp4",
    },
]


def sample_keyframes(video_path: Path, n: int = 32, target_fps: float = 10.0):
    """Decode video and return up to n PIL frames sampled at target_fps."""
    if not video_path.is_file():
        return []
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    interval = 1.0 / target_fps if target_fps > 0 else 0.0
    next_t = 0.0
    src_idx = 0
    out = []
    while True:
        ret, frame = cap.read()
        if not ret or len(out) >= n:
            break
        t = src_idx / src_fps
        src_idx += 1
        if interval == 0 or t + 1e-9 >= next_t:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(Image.fromarray(rgb))
            while interval > 0 and t + 1e-9 >= next_t:
                next_t += interval
    cap.release()
    return out


def build_prompt(four_dsg_dict, video_keyframes, question, traj_step=4):
    """Paper-spec prompt: every anchor + every keyframe.
    Trajectory uses every traj_step-th observation (default 4 → up to 8 entries
    per track for 32-frame videos), matching paper's compact F_k serialization
    that yields ~14k text tokens for K=20 (Tab. tab:speed_sg)."""
    meta = four_dsg_dict.get("metadata", {})
    guide = meta.get("reasoning_guide", "")
    n_frames = meta.get("num_frames", "?")
    n_tracks = meta.get("num_tracks", "?")
    coord = meta.get("coordinate_system", "unknown")

    crops = []
    obj_lines = []
    for track in four_dsg_dict.get("tracks", []):
        oid = track.get("object_id", "?")
        fk = track.get("F_k", [])
        ext = track.get("extent", [0, 0, 0])
        ext_s = f"{ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f}m"
        pos = track.get("image_position", "center")
        header = f"Obj{oid} obs={len(fk)} size={ext_s} img={pos}"
        traj = []
        # Stride-sampled trajectory: keeps endpoints + intermediate samples.
        sampled = fk[::max(1, traj_step)] if fk else []
        if fk and (not sampled or sampled[-1] is not fk[-1]):
            sampled = list(sampled) + [fk[-1]]
        for obs in sampled:
            t = obs.get("t", "?")
            c = obs.get("c", [])
            cs = (f"[{c[0]:.2f},{c[1]:.2f},{c[2]:.2f}]"
                  if len(c) == 3 else str(c))
            traj.append(f"  t={t}s pos={cs}")
        obj_lines.append(header + "\n" + "\n".join(traj))
        va = track.get("visual_anchor")
        if va is not None:
            p = Path(va["path"])
            if p.exists():
                try:
                    crops.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass

    text = (
        f"You are answering a multiple-choice question about a video.\n"
        f"4D Scene Graph: {n_frames} frames, {n_tracks} tracked objects, "
        f"coordinate system {coord}.\n"
        f"{guide}\n\n"
        f"Tracked objects:\n" + "\n".join(obj_lines) + "\n\n"
        f"Question: {question}\n\n"
        "You are given a 4D scene graph (4DSG) with per-object crop images, "
        "sampled video frames, and 3D tracking data. Use BOTH the visual "
        "information (video frames, object crops) and the 3D data (3D "
        "coordinates and shape stats) to reason about the answer.\n\n"
        "Answer the given multiple-choice question step by step. First, "
        "identify the relevant objects from their crop images. Then, analyze "
        "spatial/temporal relations using the 3D trajectory data. In the "
        "last sentence of your response, you must conclude by stating the "
        "final answer using the following format: 'Therefore, the final "
        "answer is: $LETTER' (without quotes), where $LETTER must be only "
        "one of the options (A or B or C or D)."
    )
    return crops, video_keyframes, text


def main():
    print("Loading Qwen2.5-VL-32B...", flush=True)
    t0 = time.time()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(QWEN_PATH),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(QWEN_PATH))
    tokenizer = processor.tokenizer
    print(f"  Qwen loaded in {time.time() - t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    results = []

    for s in SAMPLES:
        if not s["fdsg"].is_file():
            print(f"  MISSING: {s['fdsg']}", flush=True)
            continue
        if not s["video"].is_file():
            print(f"  MISSING: {s['video']}", flush=True)
            continue
        with open(s["fdsg"]) as f:
            fdsg = json.load(f)
        keyframes = sample_keyframes(s["video"], n=32, target_fps=10.0)
        n_tracks = len(fdsg["tracks"])
        n_alive_crops = sum(
            1 for t in fdsg["tracks"]
            if Path(t.get("visual_anchor", {}).get("path", "")).exists()
        )
        print(f"\n=== {s['name']} ===  tracks={n_tracks}, "
              f"crops alive={n_alive_crops}, keyframes={len(keyframes)}",
              flush=True)
        if n_alive_crops == 0:
            print("  All crops missing on disk — skipping.", flush=True)
            continue

        question = (
            "In which direction does the most prominent object move "
            "relative to the camera?\n"
            "A. left\nB. right\nC. closer\nD. farther"
        )
        crops, vframes, text = build_prompt(fdsg, keyframes, question)

        # Prepare inputs once (not part of timing — paper Tab.5 measures T_p)
        all_images = vframes + crops
        content = [{"type": "image"} for _ in all_images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt_text],
            images=all_images if all_images else None,
            padding=True, return_tensors="pt",
        ).to("cuda:0")

        n_text_tokens = int(inputs["input_ids"].shape[1])
        n_images = len(all_images)
        print(f"  prompt: text_tokens={n_text_tokens}, images={n_images} "
              f"(keyframes={len(vframes)}, crops={len(crops)})", flush=True)

        # Warmup (1)
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=512,
                               do_sample=False, use_cache=True,
                               pad_token_id=tokenizer.eos_token_id)

        # Timed trials (5)
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            t = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=512,
                                     do_sample=False, use_cache=True,
                                     pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            dt = time.time() - t
            times.append(dt)
        med = statistics.median(times)
        mn = min(times)
        mx = max(times)
        new_tokens = int(out[0, inputs["input_ids"].shape[1]:].shape[0])
        decoded = tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"  VLM: median={med:.2f}s  min={mn:.2f}s  max={mx:.2f}s  "
              f"trials={['%.2f'%x for x in times]}  "
              f"new_tokens={new_tokens}", flush=True)
        print(f"  answer (truncated): {decoded[:200]!r}", flush=True)
        results.append(dict(
            sample=s["name"], n_tracks=n_tracks, n_alive_crops=n_alive_crops,
            n_keyframes=len(vframes), n_text_tokens=n_text_tokens,
            n_images=n_images, vlm_median_s=med, vlm_min_s=mn,
            vlm_max_s=mx, trials=times, new_tokens=new_tokens,
        ))

    out = ROOT / "benchmark" / "vlm_only_timing.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
