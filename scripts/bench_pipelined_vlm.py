"""End-to-end pipelined benchmark: vision + Qwen2.5-VL-32B with constrained MC decoding.

Strategy:
  1. Load warm pool (multiplex + DA3 + FastSAM) once.
  2. Load Qwen2.5-VL-32B with sdpa attention; build a constrained-decode helper
     that emits a single A/B/C/D token (logit-mask all other tokens to -inf).
  3. Override pool._query_vlm to use the local Qwen model + force-prefix decoding.
  4. Use run_inference_pipelined to overlap GPU work for video N+1 with
     CPU 4DSG-build + VLM call for video N (VLM stays in main thread for now;
     constrained decode keeps it cheap).
  5. Report combined Hz.
"""
from __future__ import annotations

import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
os.environ["TORCH_COMPILE_SUPPRESS_ERRORS"] = "1"
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Reduce VRAM fragmentation when SAM3.1 multiplex's CUDA graph private pools
# co-tenant with Qwen-VL-32B (66 GB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
from PIL import Image

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest

VLM4D = ROOT / "benchmark" / "VLM4D-video"
HF = "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/"
QWEN_PATH = ROOT / "rose" / "models" / "qwen2.5-vl-32b-instruct"
GEMMA_PATH = ROOT / "rose" / "models" / "gemma-3-4b-it"

LETTER_CHOICES = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Qwen2.5-VL constrained MC decoder
# ---------------------------------------------------------------------------

class ConstrainedMCQwen:
    """VLM that emits one of 'A'/'B'/'C'/'D' deterministically.

    Supports two backends:
      - Qwen2.5-VL  (32B by default; 'qwen' kind)
      - Gemma 3 IT  (4B, multimodal; 'gemma' kind) — much faster forward
    """

    def __init__(self, model_path: Path, kind: str = "qwen"):
        from transformers import AutoProcessor
        from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

        self.kind = kind
        print(f"  Loading {kind} VLM from {model_path}...", flush=True)
        t0 = time.time()
        if kind == "qwen":
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        elif kind == "gemma":
            from transformers import AutoModelForImageTextToText as Cls
        else:
            raise ValueError(f"Unknown VLM kind: {kind}")
        self.model = Cls.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.tokenizer = self.processor.tokenizer
        print(f"  {kind} loaded in {time.time() - t0:.1f}s, "
              f"VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

        # Pre-compute allowed-token IDs for "A"/"B"/"C"/"D".  Different
        # tokenizers handle leading-space differently; gather both forms.
        allowed_ids = set()
        for L in LETTER_CHOICES:
            for variant in (L, " " + L):
                ids = self.tokenizer(variant, add_special_tokens=False)["input_ids"]
                if len(ids) >= 1:
                    allowed_ids.add(ids[0])
        self.allowed_ids = sorted(allowed_ids)
        print(f"  Constrained-decode token IDs (A/B/C/D variants): {self.allowed_ids}",
              flush=True)
        # Map back from token id → letter
        self._id_to_letter = {}
        for L in LETTER_CHOICES:
            for variant in (L, " " + L):
                ids = self.tokenizer(variant, add_special_tokens=False)["input_ids"]
                if len(ids) >= 1:
                    self._id_to_letter[ids[0]] = L

        class _MaskNonABCD(LogitsProcessor):
            def __init__(self, allowed):
                self.allowed = torch.tensor(allowed, dtype=torch.long)
            def __call__(self, input_ids, scores):
                mask = torch.full_like(scores, float("-inf"))
                allowed = self.allowed.to(scores.device)
                mask[:, allowed] = scores[:, allowed]
                return mask

        self._logits_proc = LogitsProcessorList([_MaskNonABCD(self.allowed_ids)])

    def query(self, four_dsg_dict, question: str, video_frames: Optional[List[Path]] = None,
              max_crops: int = 2, max_video_frames: int = 0) -> str:
        """Build a compact MC prompt and emit the chosen letter.

        Caps the number of attached images so that prompt size stays bounded:
        Qwen-VL forward latency is dominated by visual-token count, and 4DSGs
        with 13 tracks explode the prompt.  Subsample objects by F_k length
        (longer-tracked objects matter more for spatial reasoning).
        """
        # Build text content
        meta = four_dsg_dict.get("metadata", {})
        guide = meta.get("reasoning_guide", "")
        n_frames = meta.get("num_frames", "?")
        n_tracks = meta.get("num_tracks", "?")
        coord = meta.get("coordinate_system", "unknown")

        # Build per-object compact text + image refs.  Sort by len(F_k) desc
        # and keep the top N (most-tracked = most important for reasoning).
        crops: List[Image.Image] = []
        obj_lines: List[str] = []
        tracks = sorted(
            four_dsg_dict.get("tracks", []),
            key=lambda t: -len(t.get("F_k", [])),
        )[:max_crops]
        for track in tracks:
            oid = track.get("object_id", "?")
            fk = track.get("F_k", [])
            theta = track.get("theta", [0, 0])
            extent = track.get("extent", [0, 0, 0])
            extent_str = f"{extent[0]:.2f}x{extent[1]:.2f}x{extent[2]:.2f}m"
            img_pos = track.get("image_position", "center")
            motion = track.get("motion", "")
            header = (f"Obj{oid} t={theta[0]}-{theta[1]}s obs={len(fk)} "
                      f"size={extent_str} img={img_pos}")
            if motion:
                header += f" motion={motion}"
            samples = []
            # Subsample observations to 3 per track to keep prompt short
            step = max(1, len(fk) // 3)
            for obs in fk[::step][:3]:
                t = obs.get("t", "?")
                c = obs.get("c", [])
                pos_str = (f"[{c[0]:.2f},{c[1]:.2f},{c[2]:.2f}]"
                           if len(c) == 3 else str(c))
                samples.append(f"  t={t}s pos={pos_str}")
            obj_lines.append(header + "\n" + "\n".join(samples))

            # Object crop image
            va = track.get("visual_anchor")
            if va is not None:
                p = Path(va["path"])
                if p.exists():
                    try:
                        crops.append(Image.open(p).convert("RGB"))
                    except Exception:
                        pass

        # Add up to max_video_frames sampled frames for visual context.
        vframe_imgs: List[Image.Image] = []
        if video_frames and max_video_frames > 0:
            picks = video_frames[::max(1, len(video_frames)//max_video_frames)][:max_video_frames]
            for vf in picks:
                try:
                    vframe_imgs.append(Image.open(vf).convert("RGB"))
                except Exception:
                    pass

        # Build prompt content (Qwen chat template style)
        text_block = (
            f"You are answering a multiple-choice question about a video.\n"
            f"4D Scene Graph: {n_frames} frames, {n_tracks} tracked objects, "
            f"coordinate system {coord}.\n"
            f"{guide}\n\n"
            f"Tracked objects:\n" + "\n".join(obj_lines) + "\n\n"
            f"Question: {question}\n\n"
            "Reply with only one of the letters A, B, C, or D. Do NOT add "
            "any other text. Your answer:"
        )

        content = []
        # Interleave object crops and video frames as image inputs.  We'll
        # inject one <image> token per attached image at the front.
        all_images = vframe_imgs + crops
        for _ in all_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text_block})

        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt_text], images=all_images if all_images else None,
            padding=True, return_tensors="pt",
        ).to("cuda:0")

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1,  # constrained decode only emits one of A/B/C/D
                do_sample=False,
                logits_processor=self._logits_proc,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the single generated token → letter
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        if len(new_tokens) == 0:
            return "A"
        tok_id = int(new_tokens[0].item())
        if tok_id in self._id_to_letter:
            return self._id_to_letter[tok_id]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        for ch in text:
            up = ch.upper()
            if up in ("A", "B", "C", "D"):
                return up
        return "A"


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="videos to measure")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["main", "bg", "sequential"], default="main",
                    help="VLM thread placement: main (pipelined+VLM-on-main), "
                         "bg (pipelined+VLM-on-bg, experimental), or sequential")
    ap.add_argument("--no-vlm-warmup", action="store_true")
    ap.add_argument("--warmup-videos", type=int, default=15,
                    help="run all measured videos once first to populate "
                         "shape compile cache (cost paid once per process)")
    ap.add_argument("--compile", action="store_true",
                    help="enable SAM3 torch.compile (paying ~1h cold compile, "
                         "but 2-3x faster per video).  Default: eager.")
    ap.add_argument("--no-vlm", action="store_true",
                    help="skip VLM entirely — pipeline-only Hz baseline")
    ap.add_argument("--vlm", choices=["qwen", "gemma"], default="qwen",
                    help="VLM backend: qwen (Qwen2.5-VL-32B) or gemma (Gemma-3-4B-IT)")
    ap.add_argument("--max-tracks", type=int, default=50,
                    help="cap multiplex object count (smaller = faster).")
    ap.add_argument("--num-maskmem", type=int, default=7,
                    help="memory bank size for tracker (default 7; "
                         "smaller = faster cross-attn but shorter context).")
    args = ap.parse_args()

    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.offload_state_to_cpu = False
    cfg.sam3.offload_video_to_cpu = False
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = bool(args.compile)
    cfg.sam3.anchor_stride = 4
    cfg.sam3.max_active_tracks = int(args.max_tracks)
    cfg.sam3.num_maskmem = int(args.num_maskmem)
    cfg.sampling.max_frames = 32
    cfg.sampling.target_fps = 10.0

    print("=" * 72, flush=True)
    print(f"STEP 1: Load warm pool (enable_compile={cfg.sam3.enable_compile})",
          flush=True)
    print("=" * 72, flush=True)
    t0 = time.time()
    pool = WarmModelPool(cfg)
    pool.load_all()
    print(f"  load_all: {time.time() - t0:.1f}s", flush=True)

    print("\nSTEP 2: warmup_cuda (DA3 batch sizes only)", flush=True)
    t0 = time.time()
    # NOTE: SAM3 multiplex compile warmup happens inside load_all() via
    # build_sam3_multiplex_video_predictor(warm_up=True).  Calling
    # pool.warmup_compile() again would re-run the same forward sweep —
    # not free, since dynamo retraces fresh per process.  Skip it.
    pool.warmup_cuda()
    pool._status = "ready"
    print(f"  warmup total: {time.time() - t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # ------- Build sample list -------
    print("\nSTEP 3: Building sample list", flush=True)
    rng = random.Random(args.seed)
    qa_records = []
    for f in ["mini_real_mc.json", "mini_synthetic_mc.json"]:
        with open(VLM4D / "QA" / f) as fh:
            for q in json.load(fh):
                p = VLM4D / q["video"].replace(HF, "")
                if p.is_file():
                    qa_records.append({"path": p, "qa": q})

    # Dedup by path; keep one QA each
    seen = set()
    unique = []
    for r in qa_records:
        if r["path"] not in seen:
            seen.add(r["path"])
            unique.append(r)

    # Round-1 warmup uses the SAME video set as round-2 measurement, so every
    # shape encountered in measurement is already compiled.  Two rounds total.
    sample = rng.sample(unique, args.n)
    measured = sample
    throwaway = sample[:args.warmup_videos] if args.warmup_videos > 0 else []
    print(f"  sample size: {len(sample)} (warmup-round={len(throwaway)}, "
          f"measured={len(measured)})", flush=True)

    # ------- Load VLM (optional) -------
    if args.no_vlm:
        print("\nSTEP 4: Skipping VLM (pipeline-only mode)", flush=True)
    else:
        vlm_path = GEMMA_PATH if args.vlm == "gemma" else QWEN_PATH
        print(f"\nSTEP 4: Load {args.vlm}-VL (constrained MC decoder)", flush=True)
        qwen = ConstrainedMCQwen(vlm_path, kind=args.vlm)

        def _local_query_vlm(four_dsg_dict, question):
            return qwen.query(four_dsg_dict, question, video_frames=None)
        pool._query_vlm = _local_query_vlm

        if not args.no_vlm_warmup:
            print("\nSTEP 4a: Qwen warmup call", flush=True)
            t = time.time()
            warm_fdsg = {"metadata": {"num_frames": 1, "num_tracks": 0,
                                      "reasoning_guide": "", "coordinate_system": "X right Y down Z forward"},
                         "tracks": []}
            ans = qwen.query(warm_fdsg, "Test question. A B C D — answer letter:")
            print(f"  warmup VLM: {time.time() - t:.2f}s answer={ans!r}", flush=True)

    # Untimed throwaway pass — primarily useful when args.compile is on (to
    # warm shape-specific kernels).  In eager mode it's just a sanity pass.
    if args.warmup_videos > 0:
        print(f"\nSTEP 5: Throwaway pass ({args.warmup_videos} sequential, untimed)",
              flush=True)
        for r in throwaway:
            t = time.time()
            req = InferenceRequest(
                video_path=str(r["path"]),
                question=None if args.no_vlm
                                else _format_mc_question(r["qa"]),
            )
            resp = pool.run_inference(req)
            print(f"  throwaway {r['path'].name}: {time.time() - t:.2f}s "
                  f"status={resp.status} answer={resp.answer!r}", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("STEP 6: Mini-200 measurement", flush=True)
    print("=" * 72, flush=True)
    requests = [
        InferenceRequest(video_path=str(r["path"]),
                         question=None if args.no_vlm
                                  else _format_mc_question(r["qa"]))
        for r in measured
    ]
    gold = [r["qa"]["answer"] for r in measured]
    choice_letters = [_choice_letter(r["qa"]) for r in measured]

    t0 = time.time()
    if args.mode == "main":
        responses = pool.run_inference_pipelined(requests)
    elif args.mode == "bg":
        responses = _run_pipelined_bg(pool, requests)
    else:
        # sequential: full per-video, no overlap.  Used to isolate hangs
        # observed in pipelined mode.
        responses = [pool.run_inference(req) for req in requests]
    wall = time.time() - t0

    rows = []
    correct = 0
    for i, (req, resp, r, gold_letter) in enumerate(zip(requests, responses, measured, choice_letters), 1):
        if resp.status != "ok":
            print(f"  [{i:>2}] {Path(req.video_path).name}: FAILED "
                  f"{resp.error_message!r}", flush=True)
            continue
        md = (resp.four_dsg_dict or {}).get("metadata", {})
        n = md.get("num_frames", 0)
        t = md.get("num_tracks", 0)
        ans = (resp.answer or "?").strip()
        ok = "✓" if ans == gold_letter else "✗"
        if ans == gold_letter:
            correct += 1
        # Diagnostic: per-track observation counts (4DSG quality check)
        tracks = (resp.four_dsg_dict or {}).get("tracks", [])
        obs_counts = [len(tr.get("F_k", [])) for tr in tracks]
        obs_summary = (f"obs={'/'.join(map(str, obs_counts))}"
                       if obs_counts else "obs=-")
        print(f"  [{i:>2}] {Path(req.video_path).name:30s} frames={n:>2} "
              f"tracks={t:>2} {obs_summary} pred={ans} gold={gold_letter} {ok}",
              flush=True)
        rows.append({"n": n, "t": t, "ans": ans, "gold": gold_letter})

    print("\n=== Pipelined + Qwen-VL Combined ===", flush=True)
    n_ok = len(rows)
    if n_ok > 0:
        total_frames = sum(r["n"] for r in rows)
        print(f"  videos OK: {n_ok}/{len(measured)}", flush=True)
        print(f"  total wall: {wall:.2f}s", flush=True)
        print(f"  total frames: {total_frames}", flush=True)
        print(f"  Effective Hz (frames/wall): {total_frames / wall:.2f}", flush=True)
        print(f"  Per-video avg: {wall / n_ok:.2f}s", flush=True)
        print(f"  MC accuracy: {correct}/{n_ok} = {100*correct/n_ok:.1f}%", flush=True)


def _format_mc_question(qa):
    q = qa.get("question", "")
    choices = qa.get("choices", {})
    parts = [q]
    for L in LETTER_CHOICES:
        if L in choices:
            parts.append(f"{L}. {choices[L]}")
    return "\n".join(parts)


def _choice_letter(qa):
    """Map gold answer text → letter."""
    gold = qa.get("answer", "")
    choices = qa.get("choices", {})
    for L, txt in choices.items():
        if txt == gold:
            return L
    return gold[:1].upper() if gold else "?"


def _run_pipelined_bg(pool, requests):
    """Variant: VLM runs in bg finalize thread (overlaps with next video's GPU)."""
    import threading
    from rose.engine.server.warm_server import InferenceResponse

    results = []
    prev_state = {}

    def _finalize_worker(state):
        try:
            torch.cuda.set_device(0)
            fdsg, sjson = pool._finalize_4dsg(
                state["pipeline"], state["best_crops"], state["frame_dir"],
            )
            state["fdsg"] = fdsg
            state["sjson"] = sjson
            if state.get("question"):
                with torch.inference_mode():
                    state["answer"] = pool._query_vlm(fdsg, state["question"])
            else:
                state["answer"] = None
        except BaseException as exc:
            state["error"] = exc

    def _drain(state):
        if "error" in state:
            return InferenceResponse(status="error", error_message=str(state["error"]),
                                     inference_time_s=round(state.get("pipeline_time", 0), 2))
        return InferenceResponse(
            status="ok", answer=state.get("answer"),
            four_dsg_dict=state["fdsg"], scene_json=state["sjson"],
            keyframe_dir=str(state["frame_dir"]),
            inference_time_s=round(state["pipeline_time"], 2),
        )

    for req in requests:
        with pool._lock:
            t0 = time.time()
            gpu_state = pool._run_gpu_phase(req)
            gpu_time = time.time() - t0
        if "error" in gpu_state:
            results.append(InferenceResponse(
                status="error", error_message=str(gpu_state["error"]),
                inference_time_s=round(gpu_time, 2),
            ))
            continue
        gpu_state["pipeline_time"] = gpu_time
        gpu_state["question"] = req.question
        cur_thread = threading.Thread(target=_finalize_worker, args=(gpu_state,), daemon=True)
        cur_thread.start()

        if "thread" in prev_state:
            prev_state["thread"].join()
            results.append(_drain(prev_state["state"]))
        prev_state = {"thread": cur_thread, "state": gpu_state}

    if "thread" in prev_state:
        prev_state["thread"].join()
        results.append(_drain(prev_state["state"]))
    return results


if __name__ == "__main__":
    main()
