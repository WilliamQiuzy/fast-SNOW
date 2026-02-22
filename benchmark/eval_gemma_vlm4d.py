#!/usr/bin/env python3
"""Evaluate gemma-3-4b-it directly on VLM4D benchmark (no Fast-SNOW pipeline).

This is a **baseline** evaluation: the VLM receives only raw video frames and
the question prompt — no 4DSG, no 3D reasoning, no STEP tokens.

Usage:
    # Local inference (default):
    python benchmark/eval_gemma_vlm4d.py \
        --qa_json benchmark/VLM4D-video/QA/real_mc.json \
        --video_root benchmark/VLM4D-video \
        --total_frames 10 --prompt cot

    # Use a different local model path:
    python benchmark/eval_gemma_vlm4d.py \
        --qa_json benchmark/VLM4D-video/QA/real_mc.json \
        --local_model /path/to/other/gemma-model

    # Google AI API inference (instead of local):
    python benchmark/eval_gemma_vlm4d.py \
        --qa_json benchmark/VLM4D-video/QA/real_mc.json \
        --use_api

    # Dry-run (no inference, just checks data loading):
    python benchmark/eval_gemma_vlm4d.py \
        --qa_json benchmark/VLM4D-video/QA/real_mc.json \
        --dry_run

Environment:
    GOOGLE_AI_API_KEY  — required only when using --use_api
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (same as VLM4D benchmark)
# ---------------------------------------------------------------------------

COT_PROMPT = Template(
    "Question: $question\n"
    "$optionized_str\n\n"
    "Answer the given multiple-choice question step by step. "
    "Begin by explaining your reasoning process clearly. "
    "In the last sentence of your response, you must conclude by "
    "stating the final answer using the following format: "
    "'Therefore, the final answer is: $$LETTER' (without quotes), "
    "where $$LETTER must be only one of the options (A or B or C or D). "
    "Think step by step before answering."
)

DO_PROMPT = Template(
    "Question: $question\n"
    "$optionized_str\n\n"
    "Do not generate any intermediate reasoning process. "
    "Answer directly with the option letter from the given choices."
)

PROMPT_MAP = {
    "cot": COT_PROMPT,
    "direct-output": DO_PROMPT,
}

# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

HF_URL_PREFIX = (
    "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/"
)


def resolve_local_video_path(
    video_url: str, video_root: Path
) -> Optional[Path]:
    """Map a HuggingFace URL to a local file path under *video_root*.

    E.g.  .../videos_real/davis/aerobatics.mp4
       → video_root / videos_real / davis / aerobatics.mp4
    """
    if video_url.startswith(HF_URL_PREFIX):
        rel = video_url[len(HF_URL_PREFIX):]
    else:
        # Try to extract the relative path from the URL
        parsed = urlparse(video_url)
        parts = parsed.path.split("/")
        # Find 'videos_real' or 'videos_synthetic' in the path
        for i, p in enumerate(parts):
            if p in ("videos_real", "videos_synthetic"):
                rel = "/".join(parts[i:])
                break
        else:
            return None

    local = video_root / rel
    return local if local.exists() else None


def extract_frames(
    video_path: Path,
    total_frames: int = 10,
) -> List[bytes]:
    """Extract *total_frames* uniformly sampled JPEG frames from a video.

    Returns a list of JPEG-encoded byte buffers.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    all_frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        raise ValueError(f"No frames decoded from {video_path}")

    n = len(all_frames)
    if total_frames >= n:
        indices = list(range(n))
    else:
        indices = np.linspace(0, n - 1, total_frames, dtype=int).tolist()

    jpeg_frames: List[bytes] = []
    for idx in indices:
        ok, buf = cv2.imencode(".jpg", all_frames[idx])
        if ok:
            jpeg_frames.append(buf.tobytes())
    return jpeg_frames


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_answer_letter(
    response: str, choices: Dict[str, str]
) -> str:
    """Extract the predicted letter (A/B/C/D) from VLM response text.

    Strategies (in priority order):
    1. 'Therefore, the final answer is: X' pattern (CoT suffix)
    2. Starts with a single letter A-D
    3. 'Answer: X' or 'option X' patterns
    4. Parenthesised letter (X)
    5. Match choice text verbatim in response
    6. First standalone A-D token
    """
    valid = {k.upper() for k in choices}
    resp = response.strip()

    # Strategy 1: CoT final answer pattern
    m = re.search(
        r"final answer is[:\s]*\$?\s*([A-D])", resp, re.IGNORECASE
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Strategy 2: response IS just a letter
    if len(resp) == 1 and resp.upper() in valid:
        return resp.upper()

    # Strategy 3: starts with letter + delimiter
    for k in valid:
        if resp.upper().startswith(k + " ") or resp.upper().startswith(k + ".") \
                or resp.upper().startswith(k + ")"):
            return k

    # Strategy 4: keyword patterns
    for pat in [
        r"answer[:\s]+([A-D])",
        r"answer\s+is[:\s]+([A-D])",
        r"option[:\s]+([A-D])",
        r"choice[:\s]+([A-D])",
    ]:
        m = re.search(pat, resp, re.IGNORECASE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    # Strategy 5: parenthesised letter
    for pat in [r"\(([A-D])\)", r"\[([A-D])\]"]:
        m = re.search(pat, resp)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    # Strategy 6: match choice VALUE text in response
    resp_lower = resp.lower()
    for k, v in choices.items():
        if str(v).lower() in resp_lower and len(str(v)) > 2:
            return k.upper()

    # Strategy 7: first standalone A-D token
    m = re.search(r"\b([A-D])\b", resp.upper())
    if m and m.group(1) in valid:
        return m.group(1)

    return ""


def answer_matches(
    prediction_letter: str,
    ground_truth_answer: Any,
    choices: Dict[str, str],
) -> bool:
    """Check if predicted letter matches the ground-truth answer.

    VLM4D stores ground truth as the **choice value** (text or int),
    not the letter key.  So we map prediction_letter → value and compare.
    """
    if not prediction_letter:
        return False

    predicted_value = choices.get(prediction_letter)
    if predicted_value is None:
        return False

    # Compare as strings (handles int vs str mismatch)
    return str(predicted_value).strip().lower() == str(ground_truth_answer).strip().lower()


# ---------------------------------------------------------------------------
# VLM inference via Google AI API
# ---------------------------------------------------------------------------


def query_gemma(
    client: Any,
    model_name: str,
    jpeg_frames: List[bytes],
    question_prompt: str,
    max_output_tokens: int = 1024,
    temperature: float = 1.0,
    max_retries: int = 5,
) -> str:
    """Send frames + question to gemma via Google Generative AI API.

    Retries with exponential backoff on 429 / RESOURCE_EXHAUSTED.
    """
    from google.genai import types

    contents: list = []

    for jpeg_data in jpeg_frames:
        contents.append(
            types.Part.from_bytes(data=jpeg_data, mime_type="image/jpeg")
        )
    contents.append(types.Part.from_text(text=question_prompt))

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                ),
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 60 * (2 ** attempt)  # 60, 120, 240 ...
                m = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
                if m:
                    wait = float(m.group(1)) + 5
                logger.warning(f"Rate limited (attempt {attempt+1}), waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries (rate limit)")


# ---------------------------------------------------------------------------
# Local inference via transformers
# ---------------------------------------------------------------------------


def load_local_model(model_path: str, quantize_4bit: bool = False) -> Tuple[Any, Any]:
    """Load Gemma-3 model + processor from a local directory.

    Args:
        model_path: Path to local model directory.
        quantize_4bit: Use 4-bit NF4 quantization to reduce VRAM usage.
            Default is False (bf16). Set True if GPU memory is limited.
    """
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    logger.info(f"Loading local model from {model_path} (4bit={quantize_4bit}) ...")

    load_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path, **load_kwargs,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("Model loaded (VRAM allocated: %.1f GB)",
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0)
    return model, processor


MAX_IMAGE_SIDE = 512  # cap per-image resolution to fit V100-32GB


def query_gemma_local(
    model: Any,
    processor: Any,
    jpeg_frames: List[bytes],
    question_prompt: str,
    max_output_tokens: int = 1024,
) -> str:
    """Run local Gemma-3 inference with images + text prompt."""
    from PIL import Image
    import io

    content: list = []
    for jpeg_data in jpeg_frames:
        img = Image.open(io.BytesIO(jpeg_data)).convert("RGB")
        # Resize to cap VRAM usage
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": question_prompt})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    logger.debug(f"Input tokens: {input_len}")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_output_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[0][input_len:]
    text = processor.decode(generated_ids, skip_special_tokens=True)

    # Free GPU memory for next sample
    del inputs, output_ids, generated_ids
    torch.cuda.empty_cache()

    return text


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """Accumulator for VLM4D evaluation metrics."""

    total: int = 0
    correct: int = 0
    failed_extract: int = 0
    api_errors: int = 0

    # Per-category breakdown
    cat_total: Dict[str, int] = field(default_factory=dict)
    cat_correct: Dict[str, int] = field(default_factory=dict)

    # Detailed predictions
    predictions: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        sample_id: str,
        category: str,
        prediction_letter: str,
        ground_truth: Any,
        is_correct: bool,
        raw_response: str = "",
    ) -> None:
        self.total += 1
        if is_correct:
            self.correct += 1
        if not prediction_letter:
            self.failed_extract += 1

        self.cat_total[category] = self.cat_total.get(category, 0) + 1
        if is_correct:
            self.cat_correct[category] = self.cat_correct.get(category, 0) + 1

        self.predictions.append({
            "id": sample_id,
            "category": category,
            "prediction": prediction_letter,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "response": raw_response,
        })

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def category_accuracy(self, cat: str) -> float:
        t = self.cat_total.get(cat, 0)
        c = self.cat_correct.get(cat, 0)
        return c / t if t > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"VLM4D Baseline Evaluation — gemma-3-4b-it",
            "=" * 60,
            f"Overall: {self.accuracy:.2%}  ({self.correct}/{self.total})",
            f"Failed extraction: {self.failed_extract}  |  API errors: {self.api_errors}",
            "",
            "Per-category:",
        ]
        for cat in sorted(self.cat_total):
            t = self.cat_total[cat]
            c = self.cat_correct.get(cat, 0)
            lines.append(f"  {cat:20s}  {c/t:.2%}  ({c}/{t})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_accuracy": self.accuracy,
            "total": self.total,
            "correct": self.correct,
            "failed_extract": self.failed_extract,
            "api_errors": self.api_errors,
            "category_accuracy": {
                cat: {
                    "accuracy": self.category_accuracy(cat),
                    "correct": self.cat_correct.get(cat, 0),
                    "total": self.cat_total[cat],
                }
                for cat in sorted(self.cat_total)
            },
        }


# ---------------------------------------------------------------------------
# VLM4D category classification
# ---------------------------------------------------------------------------

def classify_sample(sample: Dict, split: str) -> str:
    """Classify a VLM4D sample into evaluation category.

    Categories follow acc_final_statistics.py:
    - Real split: id <= 922 → 'exo-centric', id > 922 → 'ego-centric'
    - Synthetic split: answer is "no" or starts with "no " → 'false-positive',
      else → 'directional'
    """
    if split == "real":
        num = int(sample["id"].split("_")[-1])
        return "exo-centric" if num <= 922 else "ego-centric"
    else:  # synthetic
        ans = str(sample["answer"]).strip().lower()
        if ans == "no" or ans.startswith("no "):
            return "false-positive"
        return "directional"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    qa_json: Path,
    video_root: Path,
    model_name: str = "gemma-3-4b-it",
    total_frames: int = 10,
    prompt_style: str = "cot",
    max_samples: int = -1,
    output_path: Optional[Path] = None,
    dry_run: bool = False,
    rpm_limit: int = 1,
    local_model: Optional[str] = None,
) -> EvalMetrics:
    """Run VLM4D evaluation with gemma."""

    # Determine split from filename
    fname = qa_json.stem.lower()
    if "real" in fname:
        split = "real"
    elif "synth" in fname:
        split = "synthetic"
    else:
        logger.warning(f"Cannot determine split from {fname}, defaulting to 'real'")
        split = "real"

    # Load QA data
    with open(qa_json) as f:
        samples = json.load(f)
    logger.info(f"Loaded {len(samples)} samples from {qa_json} (split={split})")

    if max_samples > 0:
        samples = samples[:max_samples]
        logger.info(f"Limiting to {max_samples} samples")

    # Prompt template
    prompt_template = PROMPT_MAP.get(prompt_style)
    if prompt_template is None:
        raise ValueError(f"Unknown prompt style: {prompt_style!r}. Choose from: {list(PROMPT_MAP)}")

    # Init inference backend
    use_local = local_model is not None
    client = None
    local_m = None
    local_proc = None

    if not dry_run:
        if use_local:
            local_m, local_proc = load_local_model(local_model)
        else:
            try:
                from google import genai
            except ImportError:
                logger.error("google-genai not installed. pip install google-genai")
                sys.exit(1)

            api_key = os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("Set GOOGLE_AI_API_KEY or GEMINI_API_KEY environment variable")
                sys.exit(1)
            client = genai.Client(api_key=api_key)
            logger.info(f"Google AI client initialized, model={model_name}")

    # API requests per minute throttle (0 means no throttle)
    request_interval = 0 if local_model is not None else (60.0 / rpm_limit if rpm_limit > 0 else 0)
    last_request_time = 0.0

    metrics = EvalMetrics()

    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        video_url = sample["video"]
        question = sample["question"]
        choices = sample["choices"]
        # Ensure all choice values are strings
        choices_str = {k: str(v) for k, v in choices.items()}
        gt_answer = sample["answer"]
        category = classify_sample(sample, split)

        # Build prompt text
        optionized_str = "\n".join(
            f"{k}: {v}" for k, v in choices_str.items()
        )
        question_prompt = prompt_template.substitute(
            question=question, optionized_str=optionized_str
        )

        if dry_run:
            logger.info(f"[{i+1}/{len(samples)}] {sample_id} ({category}) — dry run skip")
            metrics.record(sample_id, category, "", gt_answer, False, "DRY_RUN")
            continue

        # Resolve local video
        local_path = resolve_local_video_path(video_url, video_root)
        if local_path is None:
            logger.warning(f"[{i+1}/{len(samples)}] {sample_id}: video not found locally, skip")
            metrics.api_errors += 1
            metrics.record(sample_id, category, "", gt_answer, False, "VIDEO_NOT_FOUND")
            continue

        try:
            if local_model is None:
                elapsed = time.time() - last_request_time
                if elapsed < request_interval:
                    sleep_s = request_interval - elapsed
                    logger.debug(f"Sleeping {sleep_s:.2f}s for API rate limiting")
                    time.sleep(sleep_s)

            # Extract frames
            jpeg_frames = extract_frames(local_path, total_frames)
            request_start = time.time()

            # Query VLM (retry on 429 is handled inside query_gemma)
            if use_local:
                raw_response = query_gemma_local(
                    local_m, local_proc, jpeg_frames, question_prompt,
                )
            else:
                raw_response = query_gemma(
                    client, model_name, jpeg_frames, question_prompt,
                )
            last_request_time = request_start

            # Extract answer
            pred_letter = extract_answer_letter(raw_response, choices_str)
            is_correct = answer_matches(pred_letter, gt_answer, choices_str)

            metrics.record(
                sample_id, category, pred_letter, gt_answer,
                is_correct, raw_response,
            )

            elapsed_s = time.time() - request_start
            status = "✓" if is_correct else "✗"
            logger.info(
                f"[{i+1}/{len(samples)}] {sample_id} ({category}) "
                f"pred={pred_letter} gt={gt_answer} {status}  ({elapsed_s:.1f}s)"
            )

        except Exception as e:
            logger.error(f"[{i+1}/{len(samples)}] {sample_id}: {e}")
            metrics.api_errors += 1
            metrics.record(sample_id, category, "", gt_answer, False, f"ERROR: {e}")

        # Periodic progress
        if (i + 1) % 50 == 0:
            logger.info(
                f"Progress: {i+1}/{len(samples)} — "
                f"running acc={metrics.accuracy:.2%}"
            )

    # Final report
    metrics.compute = True
    print("\n" + metrics.summary())

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "config": {
                "model": model_name,
                "qa_json": str(qa_json),
                "split": split,
                "total_frames": total_frames,
                "prompt_style": prompt_style,
                "max_samples": max_samples,
            },
            "metrics": metrics.to_dict(),
            "predictions": metrics.predictions,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate gemma-3-4b-it on VLM4D benchmark (baseline, no Fast-SNOW)"
    )
    parser.add_argument(
        "--qa_json", type=str, required=True,
        help="Path to VLM4D QA JSON (e.g. benchmark/VLM4D-video/QA/real_mc.json)",
    )
    parser.add_argument(
        "--video_root", type=str,
        default="benchmark/VLM4D-video",
        help="Root directory containing videos_real/ and videos_synthetic/",
    )
    parser.add_argument(
        "--model", type=str, default="gemma-3-4b-it",
        help="Model name for Google AI API (default: gemma-3-4b-it)",
    )
    parser.add_argument(
        "--total_frames", type=int, default=10,
        help="Number of uniformly sampled frames per video (default: 10)",
    )
    parser.add_argument(
        "--prompt", type=str, default="cot",
        choices=["cot", "direct-output"],
        help="Prompt style: 'cot' (chain-of-thought) or 'direct-output'",
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1,
        help="Max number of samples to evaluate (-1 = all)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="benchmark/gemma_vlm4d",
        help="Directory for output JSON results",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load data and check video paths without making API calls",
    )
    parser.add_argument(
        "--rpm_limit", type=int, default=1,
        help="Google AI API requests per minute (default: 1). Set to 9 if available.",
    )
    parser.add_argument(
        "--local_model", type=str, default="fast_snow/models/gemma-3-4b-it",
        help="Path to local Gemma model directory (default: fast_snow/models/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--use_api", action="store_true",
        help="Use Google AI API instead of local model (requires GOOGLE_AI_API_KEY)",
    )

    args = parser.parse_args()

    qa_json = Path(args.qa_json)
    video_root = Path(args.video_root)

    if not qa_json.exists():
        logger.error(f"QA JSON not found: {qa_json}")
        sys.exit(1)
    if not video_root.exists():
        logger.error(f"Video root not found: {video_root}")
        sys.exit(1)

    # Determine local_model: None means use API
    local_model = None if args.use_api else args.local_model

    # Build output filename
    split_name = qa_json.stem  # e.g. real_mc or synthetic_mc
    tag = "api" if args.use_api else "local"
    out_name = f"{args.model}_{split_name}_{args.prompt}_{args.total_frames}f_{tag}.json"
    output_path = Path(args.output_dir) / out_name

    run_evaluation(
        qa_json=qa_json,
        video_root=video_root,
        model_name=args.model,
        total_frames=args.total_frames,
        prompt_style=args.prompt,
        max_samples=args.max_samples,
        output_path=output_path,
        dry_run=args.dry_run,
        rpm_limit=args.rpm_limit,
        local_model=local_model,
    )


if __name__ == "__main__":
    main()
