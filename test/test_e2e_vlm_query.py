"""Quick e2e test: synth_003.mp4 → Fast-SNOW 4DSG → GPT-5.2 answer."""
import os
import sys
from pathlib import Path

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fast_snow.engine.config import FastSNOWConfig, VLMConfig
from fast_snow.engine.pipeline.fast_snow_e2e import FastSNOWEndToEnd

VIDEO = "benchmark/VLM4D-video/videos_synthetic/synth_003.mp4"
QUESTION = (
    "Question: What direction is the drone moving towards?\n"
    "A: left\n"
    "B: right\n"
    "C: not moving\n"
    "D: no drone there\n\n"
    "Answer the given multiple-choice question step by step. "
    "Begin by explaining your reasoning process clearly. "
    "In the last sentence of your response, you must conclude by stating "
    "the final answer using the following format: "
    "'Therefore, the final answer is: $LETTER' (without quotes), "
    "where $LETTER must be only one of the options (A or B or C or D). "
    "Think step by step before answering."
)
GROUND_TRUTH = "A"  # left

def main():
    print("=" * 60)
    print("Fast-SNOW E2E Test: synth_003.mp4 → GPT-5.2")
    print("=" * 60)

    # Config: use OpenAI GPT-5.2, full video
    from fast_snow.engine.config import SamplingConfig
    config = FastSNOWConfig(
        sampling=SamplingConfig(target_fps=10.0),
        vlm=VLMConfig(
            provider="openai",
            model="gpt-5.2",
            max_output_tokens=1024,
            temperature=1.0,
            api_key_env="OPENAI_API_KEY",
        ),
    )

    print(f"\n[Config] VLM: {config.vlm.provider} / {config.vlm.model}")
    print(f"[Config] DA3: {config.da3.model_path}")
    print(f"[Config] YOLO: {config.yolo.model_path}")
    print(f"[Config] SAM3: {config.sam3.model_path}")
    print(f"[Config] Sampling: {config.sampling.target_fps} fps, max_frames={config.sampling.max_frames}")

    print(f"\n[Video] {VIDEO}")
    print(f"[Question] What direction is the drone moving towards?")
    print(f"[Ground Truth] {GROUND_TRUTH} (left)\n")

    e2e = FastSNOWEndToEnd(config)
    result = e2e.process_video(VIDEO, QUESTION)

    print("\n" + "=" * 60)
    print("[4DSG JSON sent to VLM]")
    print(result.scene_json)

    print("\n" + "=" * 60)
    print("[VLM Answer]")
    print(result.answer)
    print(f"\n[Ground Truth] {GROUND_TRUTH} (left)")
    correct = GROUND_TRUTH.lower() in result.answer.lower()
    print(f"[Correct] {correct}")
    print("=" * 60)

    result.cleanup()

if __name__ == "__main__":
    main()
