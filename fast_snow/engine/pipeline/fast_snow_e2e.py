"""End-to-end Fast-SNOW orchestrator.

This module wraps the vision models (DA3, FastSAM, SAM3) and feeds
their per-frame outputs into the model-agnostic FastSNOWPipeline.

Architecture:
    Phase 1 (batch):  DA3.infer_batch(all_frames) → depth/K/T_wc
                      with inter-frame consistent poses.
    Phase 2 (two-pass GPU):
        2a: FastSAM frame 0 → SAM3 init + full propagation (cache all frames)
        2b: FastSAM per-frame → discover new objects via IoU comparison
        2c: SAM3 partial propagation for new objects (merge with cached)
        2d: Build FastFrameInput per frame → CPU worker thread

Usage:
    config = FastSNOWConfig()
    e2e = FastSNOWEndToEnd(config)
    result = e2e.process_video("video.mp4", "What is in front of the car?")
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_pipeline import (
    FastSNOWPipeline,
    FastFrameInput,
    FastLocalDetection,
)
from fast_snow.vision.perception.da3_wrapper import DA3Wrapper
from fast_snow.vision.perception.fastsam_wrapper import FastSAMWrapper
from fast_snow.vision.perception.sam3_shared_session_wrapper import SAM3SharedSessionManager

logger = logging.getLogger(__name__)

_SENTINEL = None  # poison pill for CPU worker queue


# ------------------------------------------------------------------
# Two-pass discovery helpers
# ------------------------------------------------------------------

def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Return (row, col) centroid of a boolean mask."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boolean masks of the same shape."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _any_mask_iou_above(query_mask: np.ndarray, cached_masks, threshold: float) -> bool:
    """Check if query_mask overlaps with any cached SAM3 mask above threshold."""
    for m in cached_masks:
        if _mask_iou(query_mask, m.mask) >= threshold:
            return True
    return False


@dataclass
class _KeyframeDirMixin:
    """Shared cleanup logic for result objects that own a keyframe directory.

    The caller **must** call :meth:`cleanup` when the keyframe images
    referenced by ``four_dsg_dict["metadata"]["visual_anchor"]`` are no
    longer needed.  Preferred usage::

        result = e2e.process_video(...)
        try:
            ...  # use result.four_dsg_dict, result.scene_json, etc.
        finally:
            result.cleanup()
    """
    keyframe_dir: Optional[Path] = None

    def cleanup(self) -> None:
        """Remove the temporary keyframe directory."""
        if self.keyframe_dir is not None:
            shutil.rmtree(self.keyframe_dir, ignore_errors=True)
            self.keyframe_dir = None


@dataclass
class FastSNOWE2EResult(_KeyframeDirMixin):
    """End-to-end pipeline result (process_video)."""
    answer: str = ""
    scene_json: str = ""
    four_dsg_dict: Dict = field(default_factory=dict)
    step01_trace: List[Dict] = field(default_factory=list)
    keyframe_dir: Optional[Path] = None


@dataclass
class FastSNOW4DSGResult(_KeyframeDirMixin):
    """4DSG-only pipeline result (build_4dsg_from_video)."""
    four_dsg_dict: Dict = field(default_factory=dict)
    scene_json: str = ""
    keyframe_dir: Optional[Path] = None


class FastSNOWEndToEnd:
    """End-to-end orchestrator: video -> DA3/FastSAM/SAM3 -> FastSNOWPipeline -> VLM."""

    def __init__(self, config: Optional[FastSNOWConfig] = None):
        self.config = config or FastSNOWConfig()
        self._da3 = DA3Wrapper(self.config.da3)
        self._fastsam = FastSAMWrapper(self.config.fastsam)
        self._sam3 = SAM3SharedSessionManager(self.config.sam3)
        self._vlm_client = None
        # SAM3 init state (reset per video)
        self._sam3_initialized: bool = False

    def process_video(
        self,
        video_path: Union[str, Path],
        question: str,
    ) -> FastSNOWE2EResult:
        """Process video through the full Fast-SNOW pipeline.

        Phase 1: DA3 batch inference on all frames (consistent poses).
        Phase 2: Two-pass FastSAM + SAM3 (GPU) with CPU-overlap pipeline.

        Args:
            video_path: Path to MP4 video.
            question: Question to answer about the scene.

        Returns:
            FastSNOWE2EResult with answer, JSON, and 4DSG dict.
        """
        four_dsg_dict, scene_json, step01_trace, frame_dir = self._run_pipeline(video_path)
        try:
            answer = self._query_vlm(
                four_dsg_dict=four_dsg_dict,
                question=question,
            )
            return FastSNOWE2EResult(
                answer=answer,
                scene_json=scene_json,
                four_dsg_dict=four_dsg_dict,
                step01_trace=step01_trace,
                keyframe_dir=frame_dir,
            )
        except BaseException:
            shutil.rmtree(frame_dir, ignore_errors=True)
            raise

    def build_4dsg_from_video(
        self,
        video_path: Union[str, Path],
    ) -> FastSNOW4DSGResult:
        """Process video to 4DSG without VLM inference.

        Returns:
            FastSNOW4DSGResult with four_dsg_dict, scene_json, and
            keyframe_dir.  The caller **must** call ``result.cleanup()``
            when the keyframe images are no longer needed.
        """
        four_dsg_dict, scene_json, _, frame_dir = self._run_pipeline(video_path)
        return FastSNOW4DSGResult(
            four_dsg_dict=four_dsg_dict,
            scene_json=scene_json,
            keyframe_dir=frame_dir,
        )

    # ------------------------------------------------------------------
    # Internal: core pipeline (shared by process_video / build_4dsg)
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        video_path: Union[str, Path],
    ) -> Tuple[Dict, str, List[Dict], Path]:
        """Run the full 4DSG pipeline.

        Phase 1: DA3 batch on all frames → consistent depth + pose.
        Phase 2: Two-pass FastSAM + SAM3 → masks → pipeline (CPU worker).

        Returns:
            (four_dsg_dict, scene_json, step01_trace, frame_dir)
        """
        # Step 0: Extract frames
        frames, frame_dir, source_indices, keyframe_paths, timestamps_s = self._extract_frames(video_path)
        try:
            # Phase 1: DA3 batch inference — consistent inter-frame poses
            logger.info("Phase 1: DA3 batch inference on %d frames...", len(frames))
            da3_results = self._da3.infer_batch(frames)
            # Free DA3 GPU memory before loading SAM3
            self._da3.unload()

            # Phase 2: Two-pass FastSAM + SAM3
            logger.info("Phase 2: FastSAM + SAM3 two-pass on %d frames...", len(frames))
            self._sam3.set_video_dir(frame_dir)
            self._sam3_initialized = False

            step01_trace: List[Dict] = []

            # ---- Phase 2a: Init + Full Propagation ----
            fastsam_dets_0 = self._fastsam.detect(frames[0])
            cur_bboxes = [list(d.bbox_xywh_norm) for d in fastsam_dets_0]
            if cur_bboxes:
                _, init_masks = self._sam3.create_run_with_initial_bboxes(
                    boxes_xywh=cur_bboxes,
                    box_labels=[1] * len(cur_bboxes),
                    frame_idx=0,
                    tag="fastsam_bbox",
                )
                self._sam3_initialized = True
                logger.info(
                    "SAM3 initialized at frame 0 with %d FastSAM bboxes → %d masks",
                    len(cur_bboxes), len(init_masks),
                )
                # Free FastSAM GPU memory before heavy SAM3 propagation.
                # FastSAM will be lazily reloaded for Phase 2b discovery.
                self._fastsam.unload()

                # Trigger full propagation — caches ALL frames.
                # propagate_all(0) triggers propagate_in_video for all frames,
                # subsequent calls read from cache.
                for fidx in range(len(frames)):
                    self._sam3.propagate_all(fidx)
            else:
                logger.warning(
                    "FastSAM detected 0 objects on frame 0. "
                    "SAM3 not initialized — all frames will have empty masks."
                )

            # ---- Phase 2b: Discovery ----
            new_obj_count = 0
            discovery_thresh = self.config.fastsam.discovery_iou_thresh
            if self._sam3_initialized and self._sam3.active_runs:
                for fidx in range(1, len(frames)):
                    fastsam_dets = self._fastsam.detect(frames[fidx])
                    cached_masks = self._sam3.propagate_all(fidx)
                    for det in fastsam_dets:
                        if not _any_mask_iou_above(
                            det.mask, cached_masks, discovery_thresh
                        ):
                            cy, cx = _mask_centroid(det.mask)
                            self._sam3.add_object_point(fidx, (cx, cy))
                            new_obj_count += 1

            if new_obj_count > 0:
                logger.info(
                    "Discovery: found %d new objects across frames 1-%d",
                    new_obj_count, len(frames) - 1,
                )

            # Free FastSAM after discovery — not needed for propagation.
            self._fastsam.unload()

            # ---- Phase 2c: Partial Propagation ----
            if new_obj_count > 0:
                logger.info("Phase 2c: SAM3 partial propagation for %d new objects...", new_obj_count)
                self._sam3.propagate_new_objects()

            # ---- Phase 2d: Build FastFrameInputs → CPU pipeline ----
            pipeline = FastSNOWPipeline(self.config)

            cpu_queue: queue.Queue[Optional[FastFrameInput]] = queue.Queue()
            cpu_error: List[Optional[BaseException]] = [None]

            def _cpu_worker() -> None:
                try:
                    while True:
                        item = cpu_queue.get()
                        if item is _SENTINEL:
                            break
                        pipeline.process_frame(item)
                except Exception as exc:
                    cpu_error[0] = exc

            worker = threading.Thread(target=_cpu_worker, daemon=True)
            worker.start()

            try:
                for sam3_idx, image in enumerate(frames):
                    frame_masks = list(self._sam3.propagate_all(sam3_idx))
                    fi, trace = self._build_frame_input(
                        image=image,
                        da3_result=da3_results[sam3_idx],
                        sam3_frame_idx=sam3_idx,
                        source_frame_idx=source_indices[sam3_idx],
                        timestamp_s=timestamps_s[sam3_idx],
                        frame_masks=frame_masks,
                    )
                    step01_trace.append(trace)
                    cpu_queue.put(fi)
            finally:
                cpu_queue.put(_SENTINEL)
                worker.join()

            if cpu_error[0] is not None:
                raise cpu_error[0]  # type: ignore[misc]

            visual_anchor = [
                {"frame_idx": fidx, "path": str(fpath)}
                for fidx, fpath in keyframe_paths
            ]
            four_dsg_dict = pipeline.build_4dsg_dict(visual_anchor=visual_anchor)
            scene_json = json.dumps(four_dsg_dict, indent=2, sort_keys=False)
            return four_dsg_dict, scene_json, step01_trace, frame_dir
        except BaseException:
            shutil.rmtree(frame_dir, ignore_errors=True)
            raise
        finally:
            try:
                self._sam3.end_all_runs()
            except Exception:
                logger.warning("Failed to close SAM3 sessions", exc_info=True)

    # ------------------------------------------------------------------
    # Internal: frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self,
        video_path: Union[str, Path],
    ) -> Tuple[List[np.ndarray], Path, List[int], List[Tuple[int, Path]], List[float]]:
        """Extract sampled frames from video, save as JPEGs for SAM3.

        Sampling follows Step 0 config:
        - default target_fps=10.0 (10 Hz)
        - optional clip by max_frames

        The caller is responsible for cleaning up the returned frame_dir
        (via shutil.rmtree) after SAM3 sessions are closed.

        Returns:
            Tuple of (frames_rgb, frame_dir, source_indices, keyframe_paths, timestamps_s)
            where keyframe_paths is ``[(source_frame_idx, jpeg_path), ...]``
            for visual anchor / VLM multimodal input (spec §4.3, line 445),
            and timestamps_s is the physical timestamp in seconds for each frame.
        """
        import cv2

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_dir = Path(tempfile.mkdtemp(prefix="fast_snow_frames_"))
        frames: List[np.ndarray] = []
        source_indices: List[int] = []
        timestamps_s: List[float] = []
        keyframe_paths: List[Tuple[int, Path]] = []
        save_idx = 0
        src_idx = 0
        target_fps = float(self.config.sampling.target_fps)
        if target_fps <= 0:
            cap.release()
            raise ValueError(f"sampling.target_fps must be > 0, got {target_fps}")
        max_frames = self.config.sampling.max_frames
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            logger.warning(
                "Video metadata reports invalid FPS (%.3f); fallback to sampling all frames.",
                source_fps,
            )
            sample_interval_s = 0.0
            source_fps = target_fps
        else:
            sample_interval_s = 1.0 / target_fps
        next_sample_s = 0.0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            src_t_s = src_idx / source_fps
            take = sample_interval_s == 0.0 or (src_t_s + 1e-9 >= next_sample_s)
            src_idx += 1
            if not take:
                continue
            if sample_interval_s > 0.0:
                while src_t_s + 1e-9 >= next_sample_s:
                    next_sample_s += sample_interval_s
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            source_indices.append(src_idx - 1)
            timestamps_s.append(src_t_s)
            jpeg_path = frame_dir / f"{save_idx:06d}.jpg"
            cv2.imwrite(str(jpeg_path), frame_bgr)
            keyframe_paths.append((src_idx - 1, jpeg_path))
            save_idx += 1
            if max_frames is not None and save_idx >= max_frames:
                break

        cap.release()
        logger.info(
            "Extracted %d sampled frames (target_fps=%.3f, source_fps=%.3f) to %s",
            len(frames),
            target_fps,
            source_fps,
            frame_dir,
        )
        return frames, frame_dir, source_indices, keyframe_paths, timestamps_s

    # ------------------------------------------------------------------
    # Internal: per-frame vision model inference (Steps 1-3)
    # ------------------------------------------------------------------

    def _build_frame_input(
        self,
        image: np.ndarray,
        da3_result,
        sam3_frame_idx: int,
        source_frame_idx: int,
        timestamp_s: float = 0.0,
        frame_masks=None,
    ) -> Tuple[FastFrameInput, Dict]:
        """Build FastFrameInput from pre-computed SAM3 masks and DA3 results.

        In the two-pass architecture, SAM3 masks are pre-computed during
        Phases 2a-2c.  This method only converts them to FastLocalDetection
        and packages them with DA3 depth/pose into a FastFrameInput.

        Args:
            image: RGB image (H, W, 3).
            da3_result: Pre-computed DA3Result for this frame.
            sam3_frame_idx: Sequential index matching saved JPEG filenames.
            source_frame_idx: Original video frame number (used as frame_idx).
            timestamp_s: Physical timestamp in seconds from video start.
            frame_masks: Pre-computed SAM3SharedMask list for this frame.
        """
        if frame_masks is None:
            frame_masks = []

        # Deduplicate by (run_id, obj_id_local), keep highest score.
        best_by_key = {}
        for m in frame_masks:
            key = (m.run_id, m.obj_id_local)
            prev = best_by_key.get(key)
            if prev is None or m.score > prev.score:
                best_by_key[key] = m
        frame_masks = list(best_by_key.values())

        # Convert SAM3 masks to FastLocalDetection
        detections = [
            FastLocalDetection(
                run_id=m.run_id,
                local_obj_id=m.obj_id_local,
                mask=m.mask,
                score=m.score,
            )
            for m in frame_masks
        ]

        frame_input = FastFrameInput(
            frame_idx=source_frame_idx,
            depth_t=da3_result.depth,
            K_t=da3_result.K,
            T_wc_t=da3_result.T_wc,
            detections=detections,
            depth_conf_t=da3_result.depth_conf,
            depth_is_metric=da3_result.is_metric,
            timestamp_s=timestamp_s,
        )
        trace = {
            "sam3_frame_idx": sam3_frame_idx,
            "source_frame_idx": source_frame_idx,
            "frame_idx": source_frame_idx,
            "mask_count": len(frame_masks),
            "active_runs": self._sam3.num_runs,
        }
        return frame_input, trace

    # ------------------------------------------------------------------
    # Internal: VLM
    # ------------------------------------------------------------------

    def _query_vlm(
        self,
        four_dsg_dict: Dict,
        question: str,
    ) -> str:
        """Query VLM with 4DSG text + keyframe images + question.

        Prompt layout (same logical structure for both APIs):
            [4DSG text]  — metadata, ego poses, tracks (tau,c,s,theta)
            [Frame 0]    — keyframe JPEG images
            [Frame 1]
            ...
            [Question]   — user question

        Supports two providers via config.vlm.provider:
            - "openai": OpenAI API (GPT-5.2, etc.)
            - "google": Google genai API (Gemini, Gemma, etc.)
        """
        provider = self.config.vlm.provider

        if provider == "openai":
            return self._query_vlm_openai(four_dsg_dict, question)
        elif provider == "google":
            return self._query_vlm_google(four_dsg_dict, question)
        else:
            raise ValueError(f"Unsupported VLM provider: {provider!r}. Use 'openai' or 'google'.")

    def _build_4dsg_text(self, four_dsg_dict: Dict) -> str:
        """Build the 4DSG text representation shared by all providers."""
        meta = four_dsg_dict.get("metadata", {})
        grid = meta.get("grid", "16x16")
        parts: List[str] = []

        # Preamble + schema explanation
        parts.append(
            "You are a spatial reasoning agent analyzing a 4D scene graph (4DSG).\n"
            f"Grid: {grid}, "
            f"Frames: {meta.get('num_frames', '?')}, "
            f"Tracks: {meta.get('num_tracks', '?')}\n"
            f"Coordinate system: {meta.get('coordinate_system', 'unknown')}\n\n"
            "[4DSG STRUCTURE]\n"
            "The 4DSG encodes every detected object across all video frames using STEP tokens.\n"
            "Each object track contains per-frame observations (F_k) with these fields:\n"
            f"  tau  — Image patch tokens: which cells in a {grid} grid the object occupies. "
            "(row, col) = grid position, iou = how much of that cell the object covers. "
            "Tracks tau across frames to see how the object moves in the image.\n"
            "  c    — Centroid: 3D center [x, y, z] of the object in world coordinates (metres).\n"
            "  s    — Shape: per-axis Gaussian statistics (mu, sigma, min, max) of the object's 3D point cloud.\n"
            "  theta — Temporal span: [t_start, t_end] = first and last frame the object was seen.\n\n"
            "Other sections:\n"
            "  EGO POSES — Camera position [x, y, z] in world coordinates at each frame.\n"
            "  Use ego poses + object centroids (c) to derive spatial relations "
            "(distance, bearing, relative motion) yourself.\n"
        )

        # Ego poses
        ego_entries = four_dsg_dict.get("ego", [])
        if ego_entries:
            ego_lines = [f"  t={e['t']}: xyz={e['xyz']}" for e in ego_entries]
            parts.append("[EGO POSES]\n" + "\n".join(ego_lines) + "\n")

        # Per-object F_k tracks
        tracks = four_dsg_dict.get("tracks", [])
        for track in tracks:
            oid = track.get("object_id", "?")
            fk = track.get("F_k", [])
            lines = [f"=== Object {oid} (F_k, {len(fk)} frames) ==="]
            for obs in fk:
                t = obs.get("t", "?")
                c = obs.get("c", [])
                s = obs.get("s", {})
                theta = obs.get("theta", [])
                tau_grid = [(p["row"], p["col"], round(p["iou"], 2))
                            for p in obs.get("tau", [])]
                lines.append(
                    f"  t={t}: tau={tau_grid} c={c} "
                    f"s={json.dumps(s)} theta={theta}"
                )
            parts.append("\n".join(lines) + "\n")

        return "\n".join(parts)

    def _get_keyframe_paths(self, four_dsg_dict: Dict) -> List[Tuple[int, Path]]:
        """Extract valid keyframe (frame_idx, path) pairs from visual_anchor."""
        meta = four_dsg_dict.get("metadata", {})
        result = []
        for va in meta.get("visual_anchor", []):
            kf_path = Path(va["path"])
            if kf_path.exists():
                result.append((va["frame_idx"], kf_path))
        return result

    def _query_vlm_openai(self, four_dsg_dict: Dict, question: str) -> str:
        """Query VLM via OpenAI API (GPT-5.2, etc.)."""
        import base64
        import os

        if self._vlm_client is None:
            from openai import OpenAI
            api_key = os.environ.get(self.config.vlm.api_key_env)
            if not api_key:
                raise ValueError(f"{self.config.vlm.api_key_env} env var required")
            kwargs = {"api_key": api_key}
            if self.config.vlm.base_url:
                kwargs["base_url"] = self.config.vlm.base_url
            self._vlm_client = OpenAI(**kwargs)

        content: List[Dict] = []

        # 1) 4DSG text
        sg_text = self._build_4dsg_text(four_dsg_dict)
        content.append({"type": "text", "text": sg_text})

        # 2) Keyframe images
        for frame_idx, kf_path in self._get_keyframe_paths(four_dsg_dict):
            b64 = base64.b64encode(kf_path.read_bytes()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
            content.append({
                "type": "text",
                "text": f"[Keyframe t={frame_idx}]",
            })

        # 3) Question
        content.append({
            "type": "text",
            "text": (
                f"[QUERY]\n{question}\n\n"
                "You are given a 4D scene graph (4DSG) and keyframe images from a video. "
                "Use BOTH the scene graph data (3D coordinates, ego poses, object tracks) "
                "AND the visual evidence from the keyframe images to reason about the answer.\n\n"
                "Answer the given multiple-choice question step by step. "
                "First, analyze the relevant objects and their spatial/temporal changes in the scene graph. "
                "Then, cross-reference with what you observe in the keyframe images. "
                "In the last sentence of your response, you must conclude by stating the final answer "
                "using the following format: 'Therefore, the final answer is: $LETTER' (without quotes), "
                "where $LETTER must be only one of the options (A or B or C or D)."
            ),
        })

        # GPT-5.x, o1, o3, o4 use max_completion_tokens; older models use max_tokens
        model_lower = self.config.vlm.model.lower()
        use_new_param = any(p in model_lower for p in ['o1', 'o3', 'o4', 'gpt-5'])
        token_param = "max_completion_tokens" if use_new_param else "max_tokens"

        response = self._vlm_client.chat.completions.create(
            model=self.config.vlm.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.config.vlm.temperature,
            **{token_param: self.config.vlm.max_output_tokens},
        )
        return response.choices[0].message.content

    def _query_vlm_google(self, four_dsg_dict: Dict, question: str) -> str:
        """Query VLM via Google genai API (Gemini, Gemma, etc.)."""
        import os

        if self._vlm_client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError("google-genai required for provider='google'")
            api_key = os.environ.get(self.config.vlm.api_key_env)
            if not api_key:
                raise ValueError(f"{self.config.vlm.api_key_env} env var required")
            self._vlm_client = genai.Client(api_key=api_key)

        from google.genai import types

        contents: list = []

        # 1) 4DSG text
        sg_text = self._build_4dsg_text(four_dsg_dict)
        contents.append(types.Part.from_text(text=sg_text))

        # 2) Keyframe images
        for frame_idx, kf_path in self._get_keyframe_paths(four_dsg_dict):
            contents.append(types.Part.from_bytes(
                data=kf_path.read_bytes(),
                mime_type="image/jpeg",
            ))
            contents.append(types.Part.from_text(
                text=f"[Keyframe t={frame_idx}]",
            ))

        # 3) Question
        contents.append(types.Part.from_text(text=(
            f"[QUERY]\n{question}\n\n"
            "You are given a 4D scene graph (4DSG) and keyframe images from a video. "
            "Use BOTH the scene graph data (3D coordinates, ego poses, object tracks) "
            "AND the visual evidence from the keyframe images to reason about the answer.\n\n"
            "Answer the given multiple-choice question step by step. "
            "First, analyze the relevant objects and their spatial/temporal changes in the scene graph. "
            "Then, cross-reference with what you observe in the keyframe images. "
            "In the last sentence of your response, you must conclude by stating the final answer "
            "using the following format: 'Therefore, the final answer is: $LETTER' (without quotes), "
            "where $LETTER must be only one of the options (A or B or C or D)."
        )))

        response = self._vlm_client.models.generate_content(
            model=self.config.vlm.model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=self.config.vlm.max_output_tokens,
                temperature=self.config.vlm.temperature,
            ),
        )
        return response.text
