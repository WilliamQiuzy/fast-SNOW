#!/usr/bin/env python3
"""YOLO (open-vocab / fixed-class) + SAM3 multi-run benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_snow.engine.config.fast_snow_config import SAM3Config
from fast_snow.vision.perception.sam3_shared_session_wrapper import SAM3SharedSessionManager

VIDEO_DIR = PROJECT_ROOT / "assets" / "examples_videos"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "assets" / "eval_results"
MASK_OUTPUT_ROOT = PROJECT_ROOT / "assets" / "ram_masks_yolo_bbox_sam3"


def _make_cuda_sync(device: str):
    if not device.startswith("cuda"):
        return lambda: None
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except Exception:
        pass
    return lambda: None


def extract_frames_at_fps(
    video_path: Path,
    target_fps: float,
    max_duration_s: float | None = None,
) -> Tuple[float, float, List[Tuple[int, float, np.ndarray, Path]]]:
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Video reports invalid FPS={fps}: {video_path}")

    sample_interval = 1.0 / target_fps
    next_target = 0.0
    source_idx = 0
    save_idx = 0
    frames: List[Tuple[int, float, np.ndarray, Path]] = []

    if max_duration_s is not None and max_duration_s <= 0:
        raise ValueError("max_duration_s must be > 0")

    frame_dir = Path(tempfile.mkdtemp(prefix="yolo_bbox_frames_"))

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        ts = source_idx / fps
        if max_duration_s is not None and ts >= max_duration_s:
            break

        if ts >= next_target:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            jpeg_path = frame_dir / f"{save_idx:06d}.jpg"
            cv2.imwrite(str(jpeg_path), bgr)
            frames.append((source_idx, round(ts, 3), rgb, jpeg_path))
            save_idx += 1
            next_target += sample_interval

        source_idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No frame sampled from video: {video_path}")
    return fps, float(sample_interval), frames


def _safe_relpath(path: Path, base: Path) -> str:
    abs_path = path.resolve()
    try:
        return str(abs_path.relative_to(base))
    except ValueError:
        return str(abs_path)


def _draw_masks(rgb: np.ndarray, masks: List[np.ndarray], alpha: float = 0.35) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.uint8)
    h, w = canvas.shape[:2]

    for i, mask in enumerate(masks):
        if mask is None:
            continue
        m = mask.astype(np.uint8)
        if m.ndim != 2:
            continue
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        if m.max() == 0:
            continue

        rng = np.random.default_rng(i + 1)
        color = tuple(int(x) for x in rng.integers(low=0, high=256, size=3, dtype=np.uint8))

        mask_bool = m > 0
        canvas_f = canvas.astype(np.float32)
        canvas_f[mask_bool] = (0.65 * canvas_f[mask_bool] + 0.35 * np.array(color, dtype=np.float32))
        canvas = canvas_f.clip(0, 255).astype(np.uint8)

    return canvas


def _normalize_names(raw: str) -> List[str]:
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    seen = set()
    out = []
    for t in tags:
        lt = t.lower()
        if lt in seen:
            continue
        seen.add(lt)
        out.append(lt)
    return out


@dataclass
class YoloDetection:
    class_name: str
    score: float
    frame_idx: int
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh_norm: Tuple[float, float, float, float]


class YoloBBoxDetector:
    """Thin wrapper around Ultralytics detector for frame-wise bbox proposals."""

    def __init__(
        self,
        model_name_or_path: str,
        prompt_classes: Optional[List[str]] = None,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str = "cuda",
        max_det: int = 200,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.prompt_classes = prompt_classes or []
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self._model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install in snow env first: `conda run -n snow pip install ultralytics`"
            ) from exc

        self._model = YOLO(self.model_name_or_path)
        if self.prompt_classes:
            # YOLO-World exposes `set_classes`; non-open-vocab models usually ignore this.
            set_classes = getattr(self._model, "set_classes", None)
            if callable(set_classes):
                self._model.set_classes(self.prompt_classes)

    def detect(self, rgb: np.ndarray, frame_idx: int) -> List[YoloDetection]:
        if self._model is None:
            self.load()

        results = self._model(
            source=rgb,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        scores = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))
        cls_idx = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.empty((0,), dtype=int)
        names = result.names or {}

        h, w = rgb.shape[:2]
        detections: List[YoloDetection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            name = str(names.get(int(cls_idx[i]), int(cls_idx[i]) if i < len(cls_idx) else ""))
            score = float(scores[i]) if i < len(scores) else 0.0

            if self.prompt_classes and name.lower() not in self.prompt_classes:
                continue

            x1n = float(max(0.0, min(1.0, x1 / w)))
            y1n = float(max(0.0, min(1.0, y1 / h)))
            x2n = float(max(0.0, min(1.0, x2 / w)))
            y2n = float(max(0.0, min(1.0, y2 / h)))
            ww = max(0.0, x2n - x1n)
            hh = max(0.0, y2n - y1n)

            if ww <= 0 or hh <= 0:
                continue

            detections.append(
                YoloDetection(
                    class_name=name,
                    score=score,
                    frame_idx=frame_idx,
                    bbox_xyxy=(x1n, y1n, x2n, y2n),
                    bbox_xywh_norm=(x1n, y1n, ww, hh),
                )
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO bbox + SAM3 shared-session benchmark")
    parser.add_argument("--videos", nargs="*", default=None, help="Video filenames under assets/examples_videos")
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Max video duration in seconds. None = full video.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sam-score-threshold-detection",
        type=float,
        default=None,
        help="Override SAM3 mask score threshold. Use 0 for all propagation outputs.",
    )
    parser.add_argument("--output-json", default=str(EVAL_OUTPUT_DIR / "yolo_sam3_eval.json"))
    parser.add_argument("--detections-json", default=None)
    parser.add_argument("--mask-root", default=str(MASK_OUTPUT_ROOT))
    parser.add_argument(
        "--detector-weights",
        default="yolo11n.pt",
        help=(
            "Ultralytics model path/name (download on first run if not local). "
            "For open-vocab use a YOLO-World/YOLO-E checkpoint path; "
            "if unavailable, use COCO models like yolo11n.pt"
        ),
    )
    parser.add_argument(
        "--detector-prompts",
        default="",
        help=(
            "Comma-separated class filter. Empty = use all classes (recommended "
            "for fixed-class models like yolo11n). Only YOLO-World/YOLO-E models "
            "support true open-vocab prompting via set_classes()."
        ),
    )
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument(
        "--reanchor-empty-threshold",
        type=int,
        default=0,
        help=(
            "Re-anchor only after this many consecutive empty SAM propagation frames. "
            "Default 0 = disabled.  Re-anchor calls add_prompt() which triggers "
            "reset_state() inside SAM3, destroying ALL tracking memory.  Enable "
            "only for debugging (e.g. --reanchor-empty-threshold 20)."
        ),
    )
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.65)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--det-max-det", type=int, default=200)
    parser.add_argument("--max-proposals", type=int, default=80)
    parser.add_argument("--label", default="yolo11n")

    args = parser.parse_args()

    if args.videos:
        video_paths = [VIDEO_DIR / v for v in args.videos]
    else:
        video_paths = sorted(VIDEO_DIR.glob("*.mp4"))

    if args.max_proposals <= 0:
        raise ValueError("max-proposals must be > 0")

    prompt_classes = _normalize_names(args.detector_prompts)

    print("Loading detector and SAM3 (shared-session mode)...")
    detector = YoloBBoxDetector(
        model_name_or_path=args.detector_weights,
        prompt_classes=prompt_classes,
        conf=args.det_conf,
        iou=args.det_iou,
        imgsz=args.det_imgsz,
        device=args.device,
        max_det=args.det_max_det,
    )
    detector.load()

    sam_cfg = SAM3Config()
    if args.sam_score_threshold_detection is not None:
        sam_cfg.score_threshold_detection = args.sam_score_threshold_detection
    sam = SAM3SharedSessionManager(sam_cfg)
    print("Models loaded.\n")

    results = []
    mask_root = Path(args.mask_root)
    mask_root.mkdir(parents=True, exist_ok=True)
    sync = _make_cuda_sync(args.device)

    for video_path in video_paths:
        print(f"Processing {video_path.name}")
        source_fps, _, sampled = extract_frames_at_fps(video_path, args.target_fps, args.max_seconds)
        dur_str = f"first {args.max_seconds:.1f}s" if args.max_seconds else "full video"
        print(
            f"  Source FPS: {source_fps:.2f}, sampled: {len(sampled)} frames at "
            f"{args.target_fps} Hz ({dur_str})"
        )

        if args.prompt_frame < 0 or args.prompt_frame >= len(sampled):
            raise ValueError(f"prompt-frame {args.prompt_frame} is out of range for {len(sampled)} frames")

        output_dir = (mask_root / video_path.stem / args.label).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        sam.set_video_dir(Path(sampled[0][3]).parent)

        frame_infos = []
        frame_timings: List[float] = []
        det_timings: List[float] = []

        all_detections: List[Dict[str, object]] = []

        prompt_src_idx, prompt_ts, prompt_rgb, prompt_frame_path = sampled[args.prompt_frame]

        t_det0 = time.perf_counter()
        prompt_dets = detector.detect(prompt_rgb, frame_idx=args.prompt_frame)
        det_timings.append((time.perf_counter() - t_det0) * 1000.0)

        if not prompt_dets:
            print(f"WARN: no prompt detections at frame {args.prompt_frame}; results will be empty.")
            video_result = {
                "video": video_path.name,
                "mode": "yolo_bbox_sam3",
                "label": args.label,
                "detector_weight": args.detector_weights,
                "detector_prompts": prompt_classes,
                "prompt_frame": args.prompt_frame,
                "prompt_source_frame": prompt_src_idx,
                "frames": [],
                "prompt_detections": [],
                "all_detections": [],
                "detection_stats": {
                    "num_frames_detected": 0,
                    "num_total_boxes": 0,
                    "max_detections_per_frame": 0,
                },
                "mask_output_dir": _safe_relpath(output_dir, PROJECT_ROOT),
            }
            results.append(video_result)
            continue

        prompt_top = prompt_dets[: args.max_proposals]
        all_detections.extend(
            {
                "frame_idx": d.frame_idx,
                "class_name": d.class_name,
                "score": round(float(d.score), 4),
                "bbox_xyxy_norm": [round(float(x), 6) for x in d.bbox_xyxy],
                "bbox_xywh_norm": [round(float(x), 6) for x in d.bbox_xywh_norm],
            }
            for d in prompt_top
        )

        global_classes: set = set(d.class_name for d in prompt_top)
        prompt_bboxes = [list(d.bbox_xywh_norm) for d in prompt_top]
        prompt_labels = [1] * len(prompt_bboxes)
        active_prompt_bboxes = list(prompt_bboxes)

        t_init = time.perf_counter()
        init_run, init_masks = sam.create_run_with_initial_bboxes(
            boxes_xywh=prompt_bboxes,
            box_labels=prompt_labels,
            frame_idx=args.prompt_frame,
            tag="yolo_bbox",
        )
        init_prompt_ms = (time.perf_counter() - t_init) * 1000.0
        init_mask_records: List[Dict[str, object]] = [
            {
                "run_id": int(m.run_id),
                "obj_id": int(m.obj_id_local),
                "score": float(m.score),
            }
            for m in init_masks
        ]

        print(
            f"  Prompt frame {args.prompt_frame} (src {prompt_src_idx}): "
            f"det={len(prompt_top)} boxes (bbox mode), classes={sorted(global_classes)} "
            f"-> {len(init_mask_records)} init masks"
        )

        reanchor_count = 0
        sam_empty_streak = 0
        per_frame_det_counts: List[int] = []

        try:
            for i, (src_idx, ts, rgb, _jpeg) in enumerate(sampled):
                sync()
                t0 = time.perf_counter()
                propagated = sam.propagate_all(i)
                sync()
                t_sam = (time.perf_counter() - t0) * 1000.0

                frame_masks = list(propagated)
                sam_empty = len(frame_masks) == 0
                if sam_empty:
                    sam_empty_streak += 1
                else:
                    sam_empty_streak = 0
                reanchor = False
                frame_det_count = 0

                if i < args.prompt_frame:
                    # SAM3 only propagates forward from the prompt frame;
                    # skip re-anchor to preserve the initial prompt state.
                    pass
                elif i == args.prompt_frame:
                    # Merge initial masks that propagation may not yet include.
                    seen_keys = {(m.run_id, m.obj_id_local) for m in frame_masks}
                    for m in init_masks:
                        key = (m.run_id, m.obj_id_local)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            frame_masks.append(m)
                    frame_det_count = len(prompt_top)
                else:
                    # Re-detect with YOLO on current frame to discover new classes
                    # or re-anchor when propagation returns empty.
                    sync()
                    t_redet = time.perf_counter()
                    cur_dets = detector.detect(rgb, frame_idx=i)
                    sync()
                    det_timings.append((time.perf_counter() - t_redet) * 1000.0)

                    frame_det_count = len(cur_dets)
                    cur_classes = set(d.class_name for d in cur_dets)
                    cur_bboxes = [list(d.bbox_xywh_norm) for d in cur_dets]
                    global_classes.update(cur_classes)

                    # Record all detections for full-video stats.
                    all_detections.extend(
                        {
                            "frame_idx": d.frame_idx,
                            "class_name": d.class_name,
                            "score": round(float(d.score), 4),
                            "bbox_xyxy_norm": [round(float(x), 6) for x in d.bbox_xyxy],
                            "bbox_xywh_norm": [round(float(x), 6) for x in d.bbox_xywh_norm],
                        }
                        for d in cur_dets
                    )

                    # Keep fallback bboxes fresh every frame.
                    if cur_bboxes:
                        active_prompt_bboxes = cur_bboxes

                    if args.reanchor_empty_threshold <= 0:
                        needs_reprompt = False
                    else:
                        needs_reprompt = (
                            sam_empty
                            and sam_empty_streak >= args.reanchor_empty_threshold
                            and frame_det_count > 0
                        )
                    if needs_reprompt:
                        sam_empty_streak = 0
                        reanchor = True
                        reanchor_count += 1
                        reprompt_boxes = active_prompt_bboxes
                        reprompt_labels = [1] * len(reprompt_boxes)
                        _, reprompt_masks = sam.create_run_with_initial_bboxes(
                            boxes_xywh=reprompt_boxes,
                            box_labels=reprompt_labels,
                            frame_idx=i,
                            tag="yolo_bbox_reanchor",
                        )
                        frame_masks.extend(reprompt_masks)

                per_frame_det_counts.append(frame_det_count)

                # Dedupe: keep best mask per (run_id, obj_id).
                best_by_key: Dict[Tuple[int, int], object] = {}
                for m in frame_masks:
                    key = (m.run_id, m.obj_id_local)
                    prev = best_by_key.get(key)
                    if prev is None or m.score > prev.score:
                        best_by_key[key] = m

                final_masks = list(best_by_key.values())
                masks = [np.array(m.mask, copy=True) for m in final_masks]
                vis = _draw_masks(rgb, masks, alpha=0.35)

                out_path = (
                    output_dir
                    / f"{video_path.stem}_{src_idx:06d}_{uuid.uuid4().hex[:8]}.png"
                ).resolve()
                cv2.imwrite(str(out_path), vis)

                frame_timings.append(t_sam)

                flag = "R" if reanchor else (" " if i != args.prompt_frame else "P")
                frame_info = {
                    "sample_idx": i,
                    "source_frame_idx": src_idx,
                    "timestamp_sec": ts,
                    "prompt_frame": args.prompt_frame,
                    "reanchor": reanchor,
                    "yolo_classes": sorted(global_classes),
                    "num_masks": len(final_masks),
                    "sam_raw_mask_count": len(propagated),
                    "yolo_det_count": frame_det_count,
                    "active_prompt_bboxes_count": len(active_prompt_bboxes),
                    "num_init_masks": len(init_mask_records) if i == args.prompt_frame else 0,
                    "sam_propagate_ms": round(t_sam, 2),
                    "vis_path": _safe_relpath(out_path, PROJECT_ROOT),
                    "init_masks": init_mask_records if i == args.prompt_frame else [],
                    "init_prompt_ms": round(init_prompt_ms, 2) if i == args.prompt_frame else 0.0,
                }

                frame_infos.append(frame_info)

                print(
                    f"    frame {i:03d} (src {src_idx}) [{flag}] "
                    f"dets={frame_det_count:02d} "
                    f"classes={len(global_classes):02d} "
                    f"masks={len(final_masks):02d} save={out_path.name}"
                )
        finally:
            sam.end_all_runs()
            for _, _, _, fpath in sampled:
                try:
                    fpath.unlink()
                except OSError:
                    pass
            try:
                Path(sampled[0][3]).parent.rmdir()
            except OSError:
                pass

        # SAM3 propagation timing stats
        ft = np.array(frame_timings) if frame_timings else np.array([0.0])
        mask_counts = [fi["num_masks"] for fi in frame_infos]
        empty_mask_frames = sum(1 for c in mask_counts if c == 0)
        timing = {
            "sam_propagate_mean_ms": round(float(np.mean(ft)), 2),
            "sam_propagate_median_ms": round(float(np.median(ft)), 2),
            "sam_propagate_p95_ms": round(float(np.percentile(ft, 95)), 2),
            "sam_propagate_min_ms": round(float(np.min(ft)), 2),
            "sam_propagate_max_ms": round(float(np.max(ft)), 2),
            "yolo_detect_mean_ms": round(float(np.mean(det_timings)), 2) if det_timings else 0.0,
            "num_detect_calls": len(det_timings),
            "reanchor_count": reanchor_count,
            "empty_mask_frames": empty_mask_frames,
            "empty_mask_rate": round(empty_mask_frames / max(len(frame_infos), 1), 4),
        }

        # Per-frame detection counts for full-video comparison
        pfd = np.array(per_frame_det_counts) if per_frame_det_counts else np.array([0])
        detection_stats = {
            "num_frames_detected": int(np.count_nonzero(pfd)),
            "num_total_boxes": len(all_detections),
            "max_detections_per_frame": int(np.max(pfd)),
            "mean_detections_per_frame": round(float(np.mean(pfd)), 2),
            "yolo_detect_mean_ms": round(float(np.mean(det_timings)), 2) if det_timings else 0.0,
        }

        results.append(
            {
                "video": video_path.name,
                "mode": "yolo_bbox_sam3",
                "label": args.label,
                "detector_weight": args.detector_weights,
                "detector_prompts": prompt_classes,
                "prompt_frame": args.prompt_frame,
                "prompt_source_frame": prompt_src_idx,
                "prompt_timestamp_sec": prompt_ts,
                "source_fps": round(source_fps, 2),
                "sampled_fps": args.target_fps,
                "num_sampled_frames": len(frame_infos),
                "timing": timing,
                "mask_output_dir": _safe_relpath(output_dir, PROJECT_ROOT),
                "all_detections": all_detections,
                "detection_stats": detection_stats,
                "frames": frame_infos,
            }
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluation JSON to: {output_path}")

    if args.detections_json is None:
        det_out = output_path.parent / (output_path.stem + "_detections.json")
    else:
        det_out = Path(args.detections_json)
    with open(det_out, "w", encoding="utf-8") as f:
        flattened = []
        for item in results:
            for d in item["all_detections"]:
                flattened.append(
                    {
                        "video": item["video"],
                        **d,
                    }
                )
        json.dump(flattened, f, indent=2, ensure_ascii=False)
    print(f"Saved detection JSON to: {det_out}")


if __name__ == "__main__":
    main()
