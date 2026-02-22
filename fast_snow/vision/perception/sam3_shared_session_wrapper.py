"""SAM3 wrapper with a single shared predictor session.

Supports two prompt modes:
- **Bbox prompts** via ``create_run_with_initial_bboxes()`` for initial
  object discovery.  SAM3's ``add_prompt()`` with bboxes calls
  ``reset_state()``, so this must only be called once per video.
- **Point prompts** via ``add_object_point()`` for incrementally adding
  new objects mid-video without destroying existing tracker state.
  This routes to SAM3's ``add_tracker_new_points()``.

After adding new objects via point prompts, call ``propagate_new_objects()``
to trigger SAM3's ``propagation_partial`` mode which propagates only the
new objects and merges with cached results for existing objects.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from fast_snow.engine.config.fast_snow_config import SAM3Config

logger = logging.getLogger(__name__)


@dataclass
class SAM3SharedTagRun:
    """One tag prompt inside the shared SAM3 session."""

    run_id: int
    tag: str
    start_frame: int
    status: str = "created"  # created | active | ended
    last_propagated_frame: int = -1
    obj_ids: Set[int] = field(default_factory=set)


@dataclass
class SAM3SharedMask:
    """A single mask output from SAM3 shared-session mode."""

    run_id: int
    obj_id_local: int
    mask: np.ndarray  # (H, W) bool
    score: float


class SAM3SharedSessionManager:
    """Manage SAM3 bbox prompts in a single shared predictor session."""

    def __init__(self, config: Optional[SAM3Config] = None):
        self.config = config or SAM3Config()
        self._predictor = None
        self._video_dir: Optional[Path] = None
        self._session_id: Optional[str] = None
        self._session_last_propagated_frame: int = -1
        self._runs: Dict[int, SAM3SharedTagRun] = {}
        self._next_run_id: int = 0
        self._obj_id_to_run_id: Dict[int, int] = {}
        # Cache propagation results per-frame.  SAM3's action_history
        # mechanism switches to fetch-only mode after two consecutive
        # propagate_in_video calls, so we must propagate ALL remaining
        # frames in a single call and cache results here.
        self._propagation_cache: Dict[int, List[SAM3SharedMask]] = {}

    def load(self) -> None:
        """Load SAM3 predictor lazily."""
        if self._predictor is not None:
            return

        sam3_src = str(Path("fast_snow/vision/sam3").resolve())
        if sam3_src not in sys.path:
            sys.path.insert(0, sam3_src)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 requires CUDA. No CUDA device found.")

        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        checkpoint_path = self._resolve_checkpoint()

        self._predictor = Sam3VideoPredictor(
            checkpoint_path=checkpoint_path,
            bpe_path=None,
            has_presence_token=True,
            geo_encoder_use_img_cross_attn=True,
            strict_state_dict_loading=True,
            async_loading_frames=False,
            apply_temporal_disambiguation=True,
        )

        # Override threshold and memory controls.
        self._predictor.model.score_threshold_detection = (
            self.config.score_threshold_detection
        )
        self._predictor.model.trim_past_non_cond_mem_for_eval = (
            self.config.trim_past_non_cond_mem_for_eval
        )
        # Disable hotstart: the default hotstart_delay=15 buffers the
        # first 15 frames and retroactively removes objects that aren't
        # "confirmed" during that window.  With bbox prompts from YOLO
        # (already filtered for quality) this causes valid objects to be
        # dropped.  Setting hotstart_delay=0 disables the mechanism.
        self._predictor.model.hotstart_delay = 0

        logger.info(
            "SAM3 shared-session predictor loaded from %s (score_thresh=%.2f)",
            checkpoint_path or "HuggingFace",
            self.config.score_threshold_detection,
        )

    def _resolve_checkpoint(self) -> Optional[str]:
        model_path = Path(self.config.model_path)
        if model_path.is_file():
            return str(model_path.resolve())
        if model_path.is_dir():
            candidates = sorted(model_path.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .pt checkpoint files found in {model_path}. "
                    f"Place SAM3 weights (sam3.pt) in {model_path} or set "
                    f"sam3.model_path to the checkpoint file directly."
                )
            path = str(candidates[0].resolve())
            logger.info("Auto-selected SAM3 checkpoint: %s", path)
            return path
        raise FileNotFoundError(
            f"SAM3 model_path not found: {model_path}. "
            f"Download weights and place in {model_path}."
        )

    def set_video_dir(self, video_dir: Union[str, Path]) -> None:
        self._video_dir = Path(video_dir)
        # Restart the session on new video to avoid stale frame caches.
        self._close_session()
        self._session_last_propagated_frame = -1
        self._runs.clear()
        self._next_run_id = 0
        self._obj_id_to_run_id.clear()
        self._propagation_cache.clear()

    def _ensure_session_started(self) -> None:
        if self._session_id is not None:
            return
        if self._video_dir is None:
            raise RuntimeError("Call set_video_dir() before creating tags.")

        result = self._predictor.start_session(
            resource_path=str(self._video_dir),
            offload_state_to_cpu=self.config.offload_state_to_cpu,
            offload_video_to_cpu=self.config.offload_video_to_cpu,
        )
        self._session_id = result["session_id"]
        self._session_last_propagated_frame = -1
        logger.debug("Started SAM3 shared session %s", self._session_id)

    def create_run_with_initial_bboxes(
        self,
        boxes_xywh: List[List[float]],
        box_labels: Optional[List[int]],
        frame_idx: int,
        tag: str = "bbox",
    ) -> Tuple[SAM3SharedTagRun, List[SAM3SharedMask]]:
        """Create ONE combined add_prompt for multiple bboxes and return initial masks.

        SAM3's add_prompt calls reset_state() internally on every invocation.
        Pass ALL bounding boxes in a single call so reset_state fires only once
        and all detected objects are preserved in the tracker.

        Args:
            boxes_xywh: List of normalised [xmin, ymin, w, h] boxes in [0, 1].
            box_labels: Per-box labels (1 = foreground). Defaults to all-1 if None.
            frame_idx: Frame index where boxes are added.
            tag: Debug label attached to this run.
        """
        if not boxes_xywh:
            raise ValueError("boxes_xywh cannot be empty")
        self.load()
        self._ensure_session_started()

        if box_labels is None:
            box_labels = [1] * len(boxes_xywh)
        if len(boxes_xywh) != len(box_labels):
            raise ValueError("boxes_xywh and box_labels must have the same length")

        # Mirror SAM3's reset_state on our side.
        for run in self._runs.values():
            if run.status == "active":
                run.status = "ended"
        self._obj_id_to_run_id.clear()
        self._propagation_cache.clear()
        self._purge_ended_runs()

        run_id = self._next_run_id
        self._next_run_id += 1
        run = SAM3SharedTagRun(run_id=run_id, tag=tag, start_frame=frame_idx)
        self._runs[run_id] = run

        prompt_result = self._predictor.add_prompt(
            session_id=self._session_id,
            frame_idx=frame_idx,
            text="visual",
            bounding_boxes=boxes_xywh,
            bounding_box_labels=box_labels,
        )
        outputs = prompt_result.get("outputs", {})
        # Use score_threshold=0.0 for prompt-frame masks: accept all objects
        # that SAM3 detected from our bboxes, regardless of initial confidence.
        # The tracker will refine scores during propagation.
        initial_masks = self._outputs_to_masks(
            run_id=run_id,
            outputs=outputs,
            assign_new_obj_to_run=True,
            score_threshold=0.0,
        )

        run.last_propagated_frame = frame_idx
        run.status = "active"
        # After add_prompt at frame_idx the session's internal cursor is at
        # frame_idx; next propagate_all must start from frame_idx+1.
        self._session_last_propagated_frame = frame_idx
        logger.debug(
            "Batch bbox prompt at frame %d: %d boxes → %d initial masks (run %d)",
            frame_idx, len(boxes_xywh), len(initial_masks), run_id,
        )
        return run, initial_masks

    def _outputs_to_masks(
        self,
        run_id: int,
        outputs: Dict[str, np.ndarray],
        assign_new_obj_to_run: bool = False,
        require_owner_match: bool = True,
        default_run_id_when_unowned: int = -1,
        score_threshold: Optional[float] = None,
    ) -> List[SAM3SharedMask]:
        out: List[SAM3SharedMask] = []
        obj_ids = outputs.get("out_obj_ids", np.array([]))
        probs = outputs.get("out_probs", np.array([]))
        masks = outputs.get("out_binary_masks", np.array([]))
        if score_threshold is None:
            score_threshold = self.config.score_threshold_detection

        for i, obj_id in enumerate(obj_ids):
            score = float(probs[i]) if i < len(probs) else 0.0
            if score < score_threshold:
                continue
            mask = masks[i] if i < len(masks) else None
            if mask is None:
                continue

            obj_id_int = int(obj_id)
            owner_run_id = self._obj_id_to_run_id.get(obj_id_int)
            if owner_run_id is None:
                if assign_new_obj_to_run:
                    self._obj_id_to_run_id[obj_id_int] = run_id
                    owner_run_id = run_id
                    if owner_run_id in self._runs:
                        self._runs[owner_run_id].obj_ids.add(obj_id_int)
                else:
                    owner_run_id = default_run_id_when_unowned

            if require_owner_match and run_id >= 0 and owner_run_id != run_id:
                continue

            out.append(
                SAM3SharedMask(
                    run_id=owner_run_id,
                    obj_id_local=obj_id_int,
                    mask=mask.astype(bool),
                    score=score,
                )
            )
        return out

    def propagate_all(self, frame_idx: int) -> List[SAM3SharedMask]:
        """Propagate the shared session to a target frame and return frame masks.

        SAM3's ``action_history`` mechanism switches to fetch-only mode after
        two consecutive ``propagate_in_video`` calls (it assumes forward +
        backward passes are done).  To avoid this, we propagate ALL remaining
        frames in a **single** ``propagate_in_video`` call on the first
        invocation and cache every frame's results.  Subsequent calls just
        return from the cache.
        """
        # Fast path: return from cache if already propagated.
        if frame_idx in self._propagation_cache:
            return self._propagation_cache[frame_idx]

        empty: List[SAM3SharedMask] = []
        if self._predictor is None or self._session_id is None:
            return empty
        if not self._runs:
            return empty
        if frame_idx < 0:
            return empty

        start_idx = max(0, self._session_last_propagated_frame + 1)
        if start_idx > frame_idx:
            return empty

        active_run_ids = {
            run_id for run_id, run in self._runs.items() if run.status == "active"
        }
        if not active_run_ids:
            return empty

        try:
            # Propagate ALL remaining frames in one call so SAM3's
            # action_history only records a single propagation entry.
            for out in self._predictor.propagate_in_video(
                session_id=self._session_id,
                propagation_direction="forward",
                start_frame_idx=start_idx,
                max_frame_num_to_track=None,  # propagate through ALL frames
            ):
                fid = int(out.get("frame_index", -1))
                outputs = out.get("outputs", {})
                self._session_last_propagated_frame = max(
                    self._session_last_propagated_frame,
                    fid,
                )
                for run_id in active_run_ids:
                    run = self._runs.get(run_id)
                    if run is not None and run.status == "active":
                        run.last_propagated_frame = max(
                            run.last_propagated_frame, fid
                        )

                frame_masks: List[SAM3SharedMask] = []
                for m in self._outputs_to_masks(
                    run_id=-1,
                    outputs=outputs,
                    assign_new_obj_to_run=False,
                    require_owner_match=False,
                    default_run_id_when_unowned=-1,
                    score_threshold=0.0,
                ):
                    # Only keep masks owned by active runs; reject unowned
                    # objects (run_id=-1) that SAM3 auto-discovered without
                    # a YOLO bbox prompt — these hurt tracking stability.
                    if m.run_id in active_run_ids:
                        frame_masks.append(m)
                self._propagation_cache[fid] = frame_masks
        except Exception as e:
            logger.warning("SAM3 shared propagation failed: %s", e, exc_info=True)
            self._close_session()
            for run in self._runs.values():
                run.status = "ended"
            self._session_last_propagated_frame = -1
            self._obj_id_to_run_id.clear()
            self._propagation_cache.clear()
            return empty

        return self._propagation_cache.get(frame_idx, empty)

    def add_object_point(
        self,
        frame_idx: int,
        point_xy: tuple,
        label: int = 1,
    ) -> int:
        """Add a new tracked object via point prompt (non-destructive).

        Uses SAM3's ``add_tracker_new_points()`` path which appends a new
        tracker state without calling ``reset_state()``.  Existing objects
        continue to be tracked undisturbed.

        Args:
            frame_idx: Frame where the object is visible.
            point_xy: ``(x, y)`` pixel coordinates of the object center.
            label: Point label (1 = foreground, 0 = background).

        Returns:
            The obj_id assigned to the new object.
        """
        active_runs = self.active_runs
        if not active_runs:
            raise RuntimeError(
                "No active run. Call create_run_with_initial_bboxes first."
            )
        run = active_runs[0]

        # Pick the next obj_id above all known IDs.
        existing_obj_ids = set(self._obj_id_to_run_id.keys())
        new_obj_id = max(existing_obj_ids, default=-1) + 1

        self._predictor.add_prompt(
            session_id=self._session_id,
            frame_idx=frame_idx,
            points=[list(point_xy)],
            point_labels=[label],
            obj_id=new_obj_id,
        )

        # Register ownership so propagate_new_objects can include it.
        self._obj_id_to_run_id[new_obj_id] = run.run_id
        run.obj_ids.add(new_obj_id)

        logger.debug(
            "Added point prompt at frame %d: obj_id=%d, point=%s (run %d)",
            frame_idx, new_obj_id, point_xy, run.run_id,
        )
        return new_obj_id

    def propagate_new_objects(self) -> None:
        """Run partial propagation for newly added objects, update cache.

        After calling ``add_object_point()`` one or more times, invoke
        this method to propagate the new objects through all frames.
        SAM3's action_history mechanism will select ``propagation_partial``
        mode, running the tracker only for new objects and merging their
        masks with the cached results from the initial full propagation.
        """
        if self._predictor is None or self._session_id is None:
            return

        active_run_ids = {
            rid for rid, r in self._runs.items() if r.status == "active"
        }
        if not active_run_ids:
            return

        try:
            for out in self._predictor.propagate_in_video(
                session_id=self._session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=None,
            ):
                fid = int(out.get("frame_index", -1))
                outputs = out.get("outputs", {})

                frame_masks: List[SAM3SharedMask] = []
                for m in self._outputs_to_masks(
                    run_id=-1,
                    outputs=outputs,
                    assign_new_obj_to_run=True,
                    require_owner_match=False,
                    default_run_id_when_unowned=-1,
                    score_threshold=0.0,
                ):
                    if m.run_id in active_run_ids:
                        frame_masks.append(m)
                self._propagation_cache[fid] = frame_masks
        except Exception as e:
            logger.warning(
                "SAM3 partial propagation failed: %s", e, exc_info=True
            )

    def debug_runs(self) -> List[Dict[str, object]]:
        return [
            {
                "run_id": run.run_id,
                "tag": run.tag,
                "status": run.status,
                "start_frame": run.start_frame,
                "last_propagated_frame": run.last_propagated_frame,
                "obj_count": len(run.obj_ids),
                "has_session": self._session_id is not None,
            }
            for run in self._runs.values()
        ]

    @property
    def num_runs(self) -> int:
        return len(self._runs)

    @property
    def active_runs(self) -> List[SAM3SharedTagRun]:
        return [r for r in self._runs.values() if r.status == "active"]

    def _close_session(self) -> None:
        if self._predictor is None or self._session_id is None:
            return
        try:
            self._predictor.close_session(self._session_id)
        except Exception:
            pass
        self._session_id = None

    def _purge_ended_runs(self) -> None:
        """Remove ended runs from _runs to prevent unbounded accumulation."""
        ended_ids = [rid for rid, r in self._runs.items() if r.status == "ended"]
        for rid in ended_ids:
            del self._runs[rid]

    def end_all_runs(self) -> None:
        self._close_session()
        for run in self._runs.values():
            run.status = "ended"
        # Clear ownership map so stale bindings do not leak into future calls.
        self._obj_id_to_run_id.clear()
        self._propagation_cache.clear()
