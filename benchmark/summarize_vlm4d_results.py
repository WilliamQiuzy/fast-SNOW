#!/usr/bin/env python3
"""Summarize VLM4D evaluation results across real + synthetic splits.

Produces a table matching the VLM4D paper format (Table 1):
  - Real: Ego-centric / Exo-centric / Overall
  - Synthetic: Directional / False-positive / Overall
  - Combined Overall

Usage:
    python benchmark/summarize_vlm4d_results.py \
        --results_dir benchmark/gemma_vlm4d

    # Or specify files explicitly:
    python benchmark/summarize_vlm4d_results.py \
        --real benchmark/gemma_vlm4d/gemma-3-4b-it_real_mc_cot_10f.json \
        --synthetic benchmark/gemma_vlm4d/gemma-3-4b-it_synthetic_mc_cot_10f.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_predictions(path: Path) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("predictions", [])


def compute_accuracy(preds: List[Dict]) -> tuple[int, int, float]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct"))
    acc = correct / total if total > 0 else 0.0
    return correct, total, acc


def summarize(
    real_path: Optional[Path] = None,
    synthetic_path: Optional[Path] = None,
) -> str:
    lines: List[str] = []

    all_correct = 0
    all_total = 0

    # ---- Real split ----
    if real_path and real_path.exists():
        preds = load_predictions(real_path)
        ego = [p for p in preds if p["category"] == "ego-centric"]
        exo = [p for p in preds if p["category"] == "exo-centric"]

        ego_c, ego_t, ego_a = compute_accuracy(ego)
        exo_c, exo_t, exo_a = compute_accuracy(exo)
        real_c, real_t, real_a = compute_accuracy(preds)

        all_correct += real_c
        all_total += real_t

        lines.append(f"Real Split ({real_path.name})")
        lines.append("-" * 50)
        lines.append(f"  Ego-centric:   {ego_a:6.2%}  ({ego_c}/{ego_t})")
        lines.append(f"  Exo-centric:   {exo_a:6.2%}  ({exo_c}/{exo_t})")
        lines.append(f"  Real Overall:  {real_a:6.2%}  ({real_c}/{real_t})")
        lines.append("")

    # ---- Synthetic split ----
    if synthetic_path and synthetic_path.exists():
        preds = load_predictions(synthetic_path)
        direct = [p for p in preds if p["category"] == "directional"]
        fp = [p for p in preds if p["category"] == "false-positive"]

        dir_c, dir_t, dir_a = compute_accuracy(direct)
        fp_c, fp_t, fp_a = compute_accuracy(fp)
        syn_c, syn_t, syn_a = compute_accuracy(preds)

        all_correct += syn_c
        all_total += syn_t

        lines.append(f"Synthetic Split ({synthetic_path.name})")
        lines.append("-" * 50)
        lines.append(f"  Directional:   {dir_a:6.2%}  ({dir_c}/{dir_t})")
        lines.append(f"  False-positive:{fp_a:6.2%}  ({fp_c}/{fp_t})")
        lines.append(f"  Synth Overall: {syn_a:6.2%}  ({syn_c}/{syn_t})")
        lines.append("")

    # ---- Combined ----
    if all_total > 0:
        combined_a = all_correct / all_total
        lines.append("=" * 50)
        lines.append(f"  COMBINED:      {combined_a:6.2%}  ({all_correct}/{all_total})")

    return "\n".join(lines)


def find_result_files(results_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Auto-detect real and synthetic result JSONs in directory."""
    real = None
    synthetic = None
    for f in sorted(results_dir.glob("*.json")):
        name = f.name.lower()
        if "real" in name and real is None:
            real = f
        elif "synth" in name and synthetic is None:
            synthetic = f
    return real, synthetic


def main():
    parser = argparse.ArgumentParser(description="Summarize VLM4D eval results")
    parser.add_argument("--results_dir", type=str, default="benchmark/gemma_vlm4d",
                        help="Directory containing result JSONs")
    parser.add_argument("--real", type=str, default=None,
                        help="Path to real split result JSON")
    parser.add_argument("--synthetic", type=str, default=None,
                        help="Path to synthetic split result JSON")
    args = parser.parse_args()

    if args.real or args.synthetic:
        real_path = Path(args.real) if args.real else None
        synth_path = Path(args.synthetic) if args.synthetic else None
    else:
        rdir = Path(args.results_dir)
        if not rdir.exists():
            print(f"Results directory not found: {rdir}", file=sys.stderr)
            sys.exit(1)
        real_path, synth_path = find_result_files(rdir)

    if not real_path and not synth_path:
        print("No result files found.", file=sys.stderr)
        sys.exit(1)

    print(summarize(real_path, synth_path))


if __name__ == "__main__":
    main()
