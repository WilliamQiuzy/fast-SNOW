"""End-to-end Fast-SNOW orchestrator.

This module wraps the three vision models (DA3, RAM++, SAM3) and feeds
their per-frame outputs into the model-agnostic FastSNOWPipeline.

Architecture (spec §3, Step 0):
    GPU thread (main): DA3 → SAM3 propagate → RAM++ → SAM3 new runs
    CPU thread (worker): backproject + fuse + STEP + SG update
    The CPU worker processes each frame's output while the GPU thread
    has already moved on to the next frame's inference.

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
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_pipeline import (
    FastSNOWPipeline,
    FastFrameInput,
    FastLocalDetection,
)
from fast_snow.vision.perception.da3_wrapper import DA3Wrapper
from fast_snow.vision.perception.ram_wrapper import RAMPlusWrapper
from fast_snow.vision.perception.sam3_wrapper import SAM3RunManager

logger = logging.getLogger(__name__)

_SENTINEL = None  # poison pill for CPU worker queue


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
    """End-to-end orchestrator: video -> DA3/RAM++/SAM3 -> FastSNOWPipeline -> VLM."""

    def __init__(self, config: Optional[FastSNOWConfig] = None):
        self.config = config or FastSNOWConfig()
        self._da3 = DA3Wrapper(self.config.da3)
        self._ram = RAMPlusWrapper(self.config.ram_plus)
        self._sam3 = SAM3RunManager(self.config.sam3)
        self._vlm_client = None

    def process_video(
        self,
        video_path: Union[str, Path],
        question: str,
    ) -> FastSNOWE2EResult:
        """Process video through the full Fast-SNOW pipeline.

        Per-frame streaming with CPU-GPU overlap: the main thread runs GPU
        inference (Steps 1-3) while a background thread runs CPU work
        (Steps 4-7) on the previous frame's output.

        Args:
            video_path: Path to MP4 video.
            question: Question to answer about the scene.

        Returns:
            FastSNOWE2EResult with answer, JSON, and 4DSG dict.
        """
        # Step 0: Extract frames
        frames, frame_dir, source_indices, keyframe_paths = self._extract_frames(video_path)
        try:
            self._sam3.set_video_dir(frame_dir)

            # Streaming per-frame: GPU (main) + CPU (worker thread)
            global_tag_set: Set[str] = set()
            step01_trace: List[Dict] = []
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
                    # GPU: DA3 -> SAM3 propagate -> RAM++ -> SAM3 new runs
                    fi, trace = self._build_frame_input(
                        image=image,
                        sam3_frame_idx=sam3_idx,
                        source_frame_idx=source_indices[sam3_idx],
                        global_tag_set=global_tag_set,
                    )
                    step01_trace.append(trace)

                    # Hand off to CPU worker (overlaps with next frame's GPU)
                    cpu_queue.put(fi)
            finally:
                # Always signal worker to stop and wait, even if GPU loop raised
                cpu_queue.put(_SENTINEL)
                worker.join()

            if cpu_error[0] is not None:
                raise cpu_error[0]  # type: ignore[misc]

            # Step 8: Serialize + VLM
            # Build visual_anchor metadata (spec §4.3, line 445):
            # keyframe image paths + frame_idx for multimodal VLM reference.
            visual_anchor = [
                {"frame_idx": fidx, "path": str(fpath)}
                for fidx, fpath in keyframe_paths
            ]
            four_dsg_dict = pipeline.build_4dsg_dict(visual_anchor=visual_anchor)
            scene_json = json.dumps(four_dsg_dict, indent=2, sort_keys=False)
            answer = self._query_vlm(scene_json, question, keyframe_paths=keyframe_paths)

            return FastSNOWE2EResult(
                answer=answer,
                scene_json=scene_json,
                four_dsg_dict=four_dsg_dict,
                step01_trace=step01_trace,
                keyframe_dir=frame_dir,
            )
        except BaseException:
            # Only delete frame_dir on failure; on success the caller
            # owns it via result.keyframe_dir / result.cleanup().
            shutil.rmtree(frame_dir, ignore_errors=True)
            raise
        finally:
            try:
                self._sam3.end_all_runs()
            except Exception:
                logger.warning("Failed to close SAM3 sessions", exc_info=True)

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
        frames, frame_dir, source_indices, keyframe_paths = self._extract_frames(video_path)
        try:
            self._sam3.set_video_dir(frame_dir)

            global_tag_set: Set[str] = set()
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
                    fi, _ = self._build_frame_input(
                        image=image,
                        sam3_frame_idx=sam3_idx,
                        source_frame_idx=source_indices[sam3_idx],
                        global_tag_set=global_tag_set,
                    )
                    cpu_queue.put(fi)
            finally:
                # Always signal worker to stop and wait, even if GPU loop raised
                cpu_queue.put(_SENTINEL)
                worker.join()

            if cpu_error[0] is not None:
                raise cpu_error[0]  # type: ignore[misc]

            visual_anchor = [
                {"frame_idx": fidx, "path": str(fpath)}
                for fidx, fpath in keyframe_paths
            ]
            four_dsg_dict = pipeline.build_4dsg_dict(visual_anchor=visual_anchor)
            return FastSNOW4DSGResult(
                four_dsg_dict=four_dsg_dict,
                scene_json=json.dumps(four_dsg_dict, indent=2, sort_keys=False),
                keyframe_dir=frame_dir,
            )
        except BaseException:
            # Only delete frame_dir on failure; on success the caller owns it.
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
    ) -> Tuple[List[np.ndarray], Path, List[int], List[Tuple[int, Path]]]:
        """Extract sampled frames from video, save as JPEGs for SAM3.

        Sampling follows Step 0 config:
        - default target_fps=10.0 (10 Hz)
        - optional clip by max_frames

        The caller is responsible for cleaning up the returned frame_dir
        (via shutil.rmtree) after SAM3 sessions are closed.

        Returns:
            Tuple of (frames_rgb, frame_dir, source_indices, keyframe_paths)
            where keyframe_paths is ``[(source_frame_idx, jpeg_path), ...]``
            for visual anchor / VLM multimodal input (spec §4.3, line 445).
        """
        import cv2

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_dir = Path(tempfile.mkdtemp(prefix="fast_snow_frames_"))
        frames: List[np.ndarray] = []
        source_indices: List[int] = []
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
        return frames, frame_dir, source_indices, keyframe_paths

    # ------------------------------------------------------------------
    # Internal: per-frame vision model inference (Steps 1-3)
    # ------------------------------------------------------------------

    def _build_frame_input(
        self,
        image: np.ndarray,
        sam3_frame_idx: int,
        source_frame_idx: int,
        global_tag_set: Set[str],
    ) -> Tuple[FastFrameInput, Dict]:
        """Run DA3 + RAM++ + SAM3 on one frame and build FastFrameInput.

        GPU execution order per spec: DA3 -> SAM3 propagation -> RAM++.

        Args:
            image: RGB image (H, W, 3).
            sam3_frame_idx: Sequential index matching saved JPEG filenames (for SAM3).
            source_frame_idx: Original video frame number (used as frame_idx in pipeline).
            global_tag_set: Accumulated set of discovered tags (mutated in-place).
        """
        # Step 2: DA3 — depth + pose
        da3_result = self._da3.infer(image)

        # Step 3: SAM3 — propagate existing runs (uses sequential JPEG index)
        all_masks = self._sam3.propagate_all(sam3_frame_idx)

        # Step 1: RAM++ — object discovery (new tags trigger new SAM3 runs)
        tags_t_raw = self._ram.infer(image)
        tags_t = self._normalize_tags(tags_t_raw)
        new_tags = [tag for tag in tags_t if tag not in global_tag_set]
        global_tag_set.update(new_tags)

        for tag in new_tags:
            _, initial_masks = self._sam3.create_run_with_initial_masks(
                tag, sam3_frame_idx,
            )
            all_masks.extend(initial_masks)

        # Deduplicate by (run_id, obj_id_local), keep highest score.
        best_by_key = {}
        for m in all_masks:
            key = (m.run_id, m.obj_id_local)
            prev = best_by_key.get(key)
            if prev is None or m.score > prev.score:
                best_by_key[key] = m
        all_masks = list(best_by_key.values())

        # Convert SAM3 masks to FastLocalDetection
        detections = [
            FastLocalDetection(
                run_id=m.run_id,
                local_obj_id=m.obj_id_local,
                mask=m.mask,
                score=m.score,
            )
            for m in all_masks
        ]

        frame_input = FastFrameInput(
            frame_idx=source_frame_idx,
            depth_t=da3_result.depth,
            K_t=da3_result.K,
            T_wc_t=da3_result.T_wc,
            detections=detections,
            depth_conf_t=da3_result.depth_conf,
        )
        trace = {
            "sam3_frame_idx": sam3_frame_idx,
            "source_frame_idx": source_frame_idx,
            "frame_idx": source_frame_idx,
            "tags_t": tags_t,
            "new_tags_t": new_tags,
            "global_tag_count": len(global_tag_set),
            "active_runs": self._sam3.num_runs,
        }
        return frame_input, trace

    def _normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize RAM++ tags for deterministic Step 1 discovery."""
        normalized: List[str] = []
        seen: Set[str] = set()
        for tag in tags:
            t = tag.strip()
            if not t:
                continue
            if self.config.ram_plus.normalize_lowercase:
                t = t.lower()
            if self.config.ram_plus.deduplicate_tags:
                if t in seen:
                    continue
                seen.add(t)
            normalized.append(t)
        return normalized

    # ------------------------------------------------------------------
    # Internal: VLM
    # ------------------------------------------------------------------

    def _query_vlm(
        self,
        scene_json: str,
        question: str,
        keyframe_paths: Optional[List[Tuple[int, Path]]] = None,
    ) -> str:
        """Query VLM with 4DSG JSON + keyframe images + question.

        Spec §8 line 296: keyframe images are passed as independent content
        blocks alongside the JSON text (not embedded in JSON).
        ``ŷ = VLM(q | M_t, I_keyframes)`` (line 324).

        Args:
            scene_json: Serialized 4DSG JSON string.
            question: User question about the scene.
            keyframe_paths: Optional list of ``(source_frame_idx, jpeg_path)``
                for visual anchor.  Each image is sent as a separate
                ``types.Part`` content block so the VLM can correlate grid
                patches (tau) with actual pixel regions.
        """
        if self._vlm_client is None:
            import os
            try:
                from google import genai
                api_key = os.environ.get("GOOGLE_AI_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_AI_API_KEY env var required")
                self._vlm_client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("google-genai required for VLM inference")

        from google.genai import types

        text_prompt = (
            "You are a spatial reasoning assistant analyzing a 4D scene.\n\n"
            f"Scene Data (JSON):\n{scene_json}\n\n"
            "Based on the scene information above, answer the following question:\n"
            f"{question}\n\n"
            "Answer with just the letter choice (A, B, C, or D):"
        )

        # Build multimodal content: keyframe images + text prompt
        # Spec line 296: "关键帧图像作为独立 content block 传入 VLM"
        contents: List[types.Part] = []
        if keyframe_paths:
            for frame_idx, img_path in keyframe_paths:
                img_path = Path(img_path)
                if img_path.exists():
                    img_bytes = img_path.read_bytes()
                    contents.append(
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg",
                        )
                    )
        contents.append(types.Part.from_text(text=text_prompt))

        response = self._vlm_client.models.generate_content(
            model="gemma-3-4b-it",
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=256,
                temperature=0.0,
            ),
        )
        return response.text
