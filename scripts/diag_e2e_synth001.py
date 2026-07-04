"""End-to-end diagnostic for synth_001.mp4.

Outputs:
  /tmp/rose_diag/fastsam/    — FastSAM per-frame mask overlays
  /tmp/rose_diag/sam3/       — SAM3 per-frame mask overlays
  /tmp/rose_diag/4dsg.json   — Final 4DSG scene graph
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rose.engine.config.rose_config import (
    DA3Config, FastSAMConfig, ROSEConfig, SAM3Config, SamplingConfig,
)
from rose.engine.pipeline.rose_e2e import _crop_object_from_mask
from rose.engine.pipeline.rose_pipeline import (
    FastFrameInput, FastLocalDetection, ROSEPipeline,
)
from rose.vision.perception.da3_wrapper import DA3Wrapper
from rose.vision.perception.fastsam_wrapper import FastSAMWrapper
from rose.vision.perception.sam3_shared_session_wrapper import SAM3SharedSessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diag")

VIDEO = ROOT / "benchmark" / "VLM4D-video" / "videos_synthetic" / "synth_001.mp4"
OUT_DIR = ROOT / "assets" / "diag_e2e_synth001"
SKIP_VIZ = True  # Skip FastSAM/SAM3 mask visualization to save ~5s

# Distinct colours for up to 20 objects
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (0, 128, 255), (128, 0, 255), (200, 200, 200), (100, 100, 100),
]


def draw_masks_on_frame(rgb: np.ndarray, masks, labels=None, alpha=0.45) -> np.ndarray:
    """Overlay coloured masks on RGB image."""
    vis = rgb.copy()
    for i, m in enumerate(masks):
        mask_bool = m if m.dtype == bool else m > 0.5
        color = COLORS[i % len(COLORS)]
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
        # Draw contour
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)
        # Label
        if labels and i < len(labels):
            ys, xs = np.where(mask_bool)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                cv2.putText(vis, str(labels[i]), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def extract_frames(video_path: Path, target_fps: float):
    """Extract frames at target_fps. Returns (frames_rgb, source_indices, timestamps)."""
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 1.0 / target_fps
    next_t = 0.0
    frames, indices, timestamps = [], [], []
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        t = idx / src_fps
        if t + 1e-9 >= next_t:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            indices.append(idx)
            timestamps.append(t)
            while t + 1e-9 >= next_t:
                next_t += interval
        idx += 1
    cap.release()
    logger.info("Extracted %d frames at %.1f fps from %s", len(frames), target_fps, video_path.name)
    return frames, indices, timestamps


def save_frames_as_jpeg(frames, out_dir: Path):
    """Save frames as 000000.jpg, 000001.jpg, ... for SAM3."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rgb in enumerate(frames):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), bgr)
    return out_dir


def mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    import time
    import torch

    timings = {}

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for d in ["fastsam", "sam3"]:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

    # ── Config ──
    config = ROSEConfig(
        sampling=SamplingConfig(target_fps=10.0),
        da3=DA3Config(chunk_size=0, chunk_overlap=5),  # auto chunk
        fastsam=FastSAMConfig(conf_threshold=0.55, iou_threshold=0.9),
        sam3=SAM3Config(offload_state_to_cpu=False, offload_video_to_cpu=False,
                        max_discovery_total=5, max_discovery_per_frame=3,
                        chunk_size=-1, vg_stride=0),  # skip VG during propagation (no quality impact, saves ~2.8s)
    )

    # ── Step 0: Extract frames ──
    t0 = time.time()
    frames, src_indices, timestamps = extract_frames(VIDEO, 10.0)
    n = len(frames)
    logger.info("Total sampled frames: %d", n)

    # Convert frames to PIL images for direct SAM3 input (avoids JPEG round-trip).
    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames]
    frame_dir = None  # no longer needed

    timings["step0_extract"] = time.time() - t0
    logger.info("TIMING step0_extract: %.2fs", timings["step0_extract"])

    try:
        # ── Phase 1: DA3 (async — runs in background while SAM3 starts) ──
        logger.info("=== Phase 1: DA3 batch inference (async) ===")
        da3_t0 = time.time()
        da3_result_holder: dict = {}

        def _run_da3():
            try:
                da3 = DA3Wrapper(config.da3)
                da3_result_holder["results"] = da3.infer_batch(frames)
                da3.unload()
                torch.cuda.empty_cache()
                da3_result_holder["time"] = time.time() - da3_t0
                logger.info("DA3 done (async): %d results in %.2fs",
                            len(da3_result_holder["results"]), da3_result_holder["time"])
            except Exception as e:
                da3_result_holder["error"] = e
                logger.error("DA3 failed: %s", e)

        da3_thread = threading.Thread(target=_run_da3, daemon=True)
        da3_thread.start()

        # ── Phase 2a: FastSAM frame 0 → SAM3 init ──
        t0 = time.time()
        logger.info("=== Phase 2a: FastSAM frame 0 + SAM3 init ===")

        # Start SAM3 model loading in background thread.
        # SAM3 checkpoint loading is mostly CPU I/O (~13s), so it can run
        # concurrently with FastSAM GPU inference (~4s) to save wall time.
        # NOTE: SAM3's tracker __init__ enters a thread-local BFloat16 autocast
        # context (sam3_tracking_predictor.py:50-51).  After joining the load
        # thread we must re-enter autocast in the main thread.
        sam3 = SAM3SharedSessionManager(config.sam3)
        sam3_load_error = [None]
        def _load_sam3_bg():
            try:
                sam3.load()
            except Exception as e:
                sam3_load_error[0] = e
        sam3_load_thread = threading.Thread(target=_load_sam3_bg, daemon=True)
        sam3_load_thread.start()

        fastsam = FastSAMWrapper(config.fastsam)

        # Store FastSAM detections per frame for visualization
        fastsam_per_frame = {}

        dets_0 = fastsam.detect(frames[0])
        fastsam_per_frame[0] = dets_0
        _t_fastsam_detect = time.time() - t0
        logger.info("FastSAM frame 0: %d detections (load+detect %.2fs)",
                     len(dets_0), _t_fastsam_detect)

        # Wait for SAM3 model loading to finish.
        sam3_load_thread.join()
        if sam3_load_error[0] is not None:
            raise sam3_load_error[0]
        # Re-enter the BFloat16 autocast that was started in the load thread
        # (autocast is thread-local, so the main thread needs its own).
        sam3._predictor.model.tracker.bf16_context = torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
        )
        sam3._predictor.model.tracker.bf16_context.__enter__()
        _t_sam3_loaded = time.time() - t0
        logger.info("SAM3 model loaded (%.2fs elapsed, overlapped with FastSAM)", _t_sam3_loaded)

        sam3.set_video_frames(pil_frames)

        bboxes = [list(d.bbox_xywh_norm) for d in dets_0]
        if bboxes:
            _t_sam3_init_start = time.time()
            _, init_masks = sam3.create_run_with_initial_bboxes(
                boxes_xywh=bboxes,
                box_labels=[1] * len(bboxes),
                frame_idx=0,
                tag="fastsam_bbox",
            )
            _t_sam3_init = time.time() - _t_sam3_init_start
            logger.info("SAM3 init: %d masks from %d bboxes (%.2fs = session + prompt, model already loaded)",
                        len(init_masks), len(bboxes), _t_sam3_init)

            # Enable backbone cache retention so partial propagation can
            # reuse cached features instead of rerunning the full VL backbone.
            sam3._predictor.model.retain_feature_cache = True
            # Pre-compute backbone features to skip the expensive VG detector
            # forward pass during full propagation.
            VG_STRIDE = config.sam3.vg_stride
            logger.info("Pre-computing backbone features (vg_stride=%d)...", VG_STRIDE)
            import time as _time_mod
            _bb_t0 = _time_mod.time()
            sam3.precompute_backbone_features(vg_stride=VG_STRIDE)
            _bb_t1 = _time_mod.time()
            logger.info("Backbone pre-computation: %.2fs", _bb_t1 - _bb_t0)
            # Apply tracker memory settings from config
            sam3._predictor.model.tracker.num_maskmem = config.sam3.num_maskmem
            sam3._predictor.model.tracker.memory_temporal_stride_for_eval = config.sam3.memory_temporal_stride
            logger.info("SAM3 acceleration: backbone_cache_retention + num_maskmem=%d + mem_stride=%d",
                        config.sam3.num_maskmem, config.sam3.memory_temporal_stride)

            fastsam.unload()
            torch.cuda.empty_cache()

            # Profiling disabled for clean timing (adds cuda.synchronize overhead)
            # sam3._predictor.model._frame_profiling = True

            # Full propagation — must run before point prompts can be added
            logger.info("SAM3 full propagation over %d frames...", n)
            _t_prop_start = time.time()
            for fidx in range(n):
                sam3.propagate_all(fidx)
            _t_prop = time.time() - _t_prop_start
            logger.info("SAM3 propagation done (cached) — %.2fs", _t_prop)

            # Add remaining frame-0 detections as point prompts, filtering
            # out those that overlap with init_masks (SAM3's visual prompt).
            MAX_FRAME0_POINTS = 8
            frame0_point_count = 0
            if len(bboxes) > 1:
                remaining_indices = list(range(1, len(dets_0)))
                remaining_indices.sort(key=lambda i: dets_0[i].mask.sum(), reverse=True)
                for i in remaining_indices:
                    if frame0_point_count >= MAX_FRAME0_POINTS:
                        break
                    if init_masks and any(
                        mask_iou(dets_0[i].mask, m.mask) >= 0.5 for m in init_masks
                    ):
                        continue
                    bx, by, bw, bh = bboxes[i]
                    cx_norm = bx + bw / 2.0
                    cy_norm = by + bh / 2.0
                    sam3.add_object_point(0, (cx_norm, cy_norm))
                    frame0_point_count += 1
                logger.info("Added %d/%d frame-0 point prompts (filtered %d overlaps)",
                            frame0_point_count, len(remaining_indices),
                            len(remaining_indices) - frame0_point_count)

                # Propagate frame-0 point prompts NOW so discovery sees all
                # tracked objects in the cache (not just the 2-3 from visual prompt).
                if frame0_point_count > 0:
                    _t_pt_prop_start = time.time()
                    sam3.propagate_new_objects()
                    _t_pt_prop = time.time() - _t_pt_prop_start
                    logger.info("Frame-0 point prompts propagated — %.2fs", _t_pt_prop)
        else:
            logger.warning("FastSAM found 0 objects on frame 0!")
            fastsam.unload()

        timings["phase2a_sam3_init_propagate"] = time.time() - t0
        logger.info("TIMING phase2a_sam3_init+propagate: %.2fs", timings["phase2a_sam3_init_propagate"])

        # Print per-step profiling results from full propagation (if enabled)
        _pa = getattr(sam3._predictor.model, '_profile_accum', None)
        if _pa and _pa.get('n', 0) > 0:
            _n = _pa['n']
            _total_steps = sum(_pa[s] for s in ['step1','step2','step3','step4','step5'])
            logger.info("=== _det_track_one_frame PROFILE (%d frames, %.3fs total) ===", _n, _total_steps)
            for k in ['step1', 'step2', 'step3', 'step4', 'step5']:
                pct = 100.0 * _pa[k] / _total_steps if _total_steps > 0 else 0
                logger.info("  %-8s total=%.3fs  avg=%.4fs  (%.1f%%)", k, _pa[k], _pa[k] / _n, pct)
            sam3._predictor.model._profile_accum = None

        # ── Phase 2b: Discovery ──
        logger.info("=== Phase 2b: FastSAM per-frame discovery ===")
        t0 = time.time()
        discovery_thresh = config.fastsam.discovery_iou_thresh
        max_disc = config.sam3.max_discovery_per_frame
        min_mask_frac = config.sam3.discovery_min_mask_frac
        new_obj_count = 0

        max_total = config.sam3.max_discovery_total
        if sam3.active_runs:
            for fidx in range(1, n):
                if max_total > 0 and new_obj_count >= max_total:
                    break
                dets = fastsam.detect(frames[fidx])
                fastsam_per_frame[fidx] = dets
                cached = sam3.propagate_all(fidx)
                frame_disc = 0
                for det in dets:
                    if max_disc > 0 and frame_disc >= max_disc:
                        break
                    if max_total > 0 and new_obj_count >= max_total:
                        break
                    # Skip tiny detections
                    if min_mask_frac > 0:
                        dh, dw = det.mask.shape[:2]
                        if det.mask.sum() < min_mask_frac * dh * dw:
                            continue
                    if not any(mask_iou(det.mask, m.mask) >= discovery_thresh for m in cached):
                        cy_px, cx_px = mask_centroid(det.mask)
                        h, w = det.mask.shape[:2]
                        sam3.add_object_point(fidx, (cx_px / w, cy_px / h))
                        new_obj_count += 1
                        frame_disc += 1

        logger.info("Discovery: %d new objects found", new_obj_count)
        fastsam.unload()
        torch.cuda.empty_cache()
        timings["phase2b_discovery"] = time.time() - t0
        logger.info("TIMING phase2b_discovery: %.2fs", timings["phase2b_discovery"])

        # ── Phase 2c: Partial propagation for discovery objects only ──
        # Frame-0 point prompts were already propagated before discovery.
        t0 = time.time()
        if new_obj_count > 0:
            logger.info("=== Phase 2c: SAM3 partial propagation (%d discovery objects) ===",
                        new_obj_count)
            sam3.propagate_new_objects()
            logger.info("Partial propagation done")

        timings["phase2c_partial_propagate"] = time.time() - t0
        logger.info("TIMING phase2c_partial_propagate: %.2fs", timings["phase2c_partial_propagate"])

        # Release retained backbone caches to free GPU memory.
        if hasattr(sam3, '_predictor') and sam3._predictor is not None:
            model = sam3._predictor.model
            model.retain_feature_cache = False
            model.detector.retain_multigpu_buffer = False
            # Clear the multigpu_buffer and per-frame feature_cache to reclaim memory
            for sid, session_data in sam3._predictor._ALL_INFERENCE_STATES.items():
                fc = session_data["state"].get("feature_cache", {})
                mb = fc.get("multigpu_buffer", {})
                n_buf = len(mb)
                mb.clear()
                # Also clear per-frame backbone features (int keys)
                frame_keys = [k for k in fc if isinstance(k, int)]
                for k in frame_keys:
                    del fc[k]
                logger.info("Cleared retained caches: %d buffer entries, %d frame features", n_buf, len(frame_keys))
            torch.cuda.empty_cache()

        # ── Visualize FastSAM + SAM3 masks ──
        t0 = time.time()
        if SKIP_VIZ:
            logger.info("Skipping FastSAM/SAM3 visualization (SKIP_VIZ=True)")
            timings["viz_fastsam_sam3"] = 0.0
        else:
            logger.info("=== Saving FastSAM visualizations ===")
            for fidx in sorted(fastsam_per_frame.keys()):
                dets = fastsam_per_frame[fidx]
                if dets:
                    masks = [d.mask for d in dets]
                    labels = [f"d{i}" for i in range(len(dets))]
                    vis = draw_masks_on_frame(frames[fidx], masks, labels)
                else:
                    vis = frames[fidx].copy()
                bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(OUT_DIR / "fastsam" / f"frame_{fidx:04d}.jpg"), bgr)
            logger.info("Saved %d FastSAM frames to %s/fastsam/", len(fastsam_per_frame), OUT_DIR)

            logger.info("=== Saving SAM3 visualizations ===")
            for fidx in range(n):
                sam3_masks = list(sam3.propagate_all(fidx))
                if sam3_masks:
                    masks = [m.mask for m in sam3_masks]
                    labels = [f"o{m.obj_id_local}" for m in sam3_masks]
                    vis = draw_masks_on_frame(frames[fidx], masks, labels)
                else:
                    vis = frames[fidx].copy()
                bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(OUT_DIR / "sam3" / f"frame_{fidx:04d}.jpg"), bgr)
            logger.info("Saved %d SAM3 frames to %s/sam3/", n, OUT_DIR)

            timings["viz_fastsam_sam3"] = time.time() - t0
            logger.info("TIMING viz_fastsam+sam3: %.2fs", timings["viz_fastsam_sam3"])

        # ── Wait for DA3 to finish ──
        da3_thread.join()
        if "error" in da3_result_holder:
            raise da3_result_holder["error"]
        da3_results = da3_result_holder["results"]
        timings["phase1_da3"] = da3_result_holder.get("time", 0.0)
        logger.info("TIMING phase1_da3: %.2fs", timings["phase1_da3"])

        # ── Phase 2d + Phase 3: CPU pipeline → 4DSG ──
        logger.info("=== Phase 3: Building 4DSG ===")
        t0 = time.time()
        pipeline = ROSEPipeline(config)
        crop_pad = config.vlm.object_crop_padding
        crop_sz = config.vlm.object_crop_size
        best_crops = {}  # (run_id, obj_id_local) → (crop_rgb, score, src_frame_idx, brightness)
        _t_crop_total = 0.0
        _t_process_frame_total = 0.0

        for fidx in range(n):
            sam3_masks = list(sam3.propagate_all(fidx))
            # Dedup
            best = {}
            for m in sam3_masks:
                key = (m.run_id, m.obj_id_local)
                if key not in best or m.score > best[key].score:
                    best[key] = m
            sam3_masks = list(best.values())

            # Collect best crop per (run_id, obj_id_local).
            # Prefer highest score; skip crop computation if score is worse.
            _t_crop_start = time.time()
            for m in sam3_masks:
                key = (m.run_id, m.obj_id_local)
                prev = best_crops.get(key)
                if prev is not None and m.score < prev[1]:
                    continue  # Lower score — skip expensive crop computation
                crop = _crop_object_from_mask(
                    frames[fidx], m.mask, padding=crop_pad, size=crop_sz,
                )
                brightness = float(crop.mean())
                if prev is None or (m.score, brightness) > (prev[1], prev[3]):
                    best_crops[key] = (crop, m.score, src_indices[fidx], brightness)
            _t_crop_total += time.time() - _t_crop_start

            detections = [
                FastLocalDetection(
                    run_id=m.run_id,
                    local_obj_id=m.obj_id_local,
                    mask=m.mask,
                    score=m.score,
                )
                for m in sam3_masks
            ]
            fi = FastFrameInput(
                frame_idx=src_indices[fidx],
                depth_t=da3_results[fidx].depth,
                K_t=da3_results[fidx].K,
                T_wc_t=da3_results[fidx].T_wc,
                detections=detections,
                depth_conf_t=da3_results[fidx].depth_conf,
                depth_is_metric=da3_results[fidx].is_metric,
                timestamp_s=timestamps[fidx],
            )
            _t_pf_start = time.time()
            pipeline.process_frame(fi)
            _t_process_frame_total += time.time() - _t_pf_start

        # Save per-object crops, resolve global IDs
        crops_dir = OUT_DIR / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        object_crops = {}
        for key, gid in pipeline._local_to_global.items():
            if key in best_crops and gid not in object_crops:
                crop_rgb, _score, src_idx, _bright = best_crops[key]
                if crop_rgb.mean() < 1.0:
                    continue  # skip all-black / near-black crops
                crop_path = crops_dir / f"obj_{gid:04d}.jpg"
                cv2.imwrite(
                    str(crop_path),
                    cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                )
                object_crops[gid] = {
                    "path": str(crop_path),
                    "source_frame_idx": src_idx,
                }
        logger.info("Saved %d object crops to %s", len(object_crops), crops_dir)

        # Post-hoc dedup: merge re-tracked objects with similar crops & centroids.
        n_before = len(object_crops)
        _t_merge_start = time.time()
        object_crops = pipeline.merge_duplicate_tracks(object_crops)
        _t_merge = time.time() - _t_merge_start
        n_after = len(object_crops)
        if n_before != n_after:
            logger.info("Dedup: %d tracks → %d tracks (merged %d duplicates)", n_before, n_after, n_before - n_after)

        _t_build_start = time.time()
        dsg = pipeline.build_4dsg_dict(object_crops=object_crops)
        _t_build = time.time() - _t_build_start

        logger.info("TIMING phase3_crop_computation:     %.4fs", _t_crop_total)
        logger.info("TIMING phase3_process_frame:        %.4fs", _t_process_frame_total)
        logger.info("TIMING phase3_merge_duplicate:      %.4fs", _t_merge)
        logger.info("TIMING phase3_build_4dsg_dict:      %.4fs", _t_build)
        _t_phase3_accounted = _t_crop_total + _t_process_frame_total + _t_merge + _t_build
        _t_phase3_total = time.time() - t0
        logger.info("TIMING phase3_accounted_TOTAL:      %.4fs / %.4fs (%.1f%%)",
                     _t_phase3_accounted, _t_phase3_total,
                     100.0 * _t_phase3_accounted / max(_t_phase3_total, 1e-9))

        dsg_json = json.dumps(dsg, indent=2, ensure_ascii=False, sort_keys=False)

        out_path = OUT_DIR / "4dsg.json"
        out_path.write_text(dsg_json)
        logger.info("4DSG saved to %s", out_path)

        # Summary
        meta = dsg.get("metadata", {})
        tracks = dsg.get("tracks", [])
        logger.info("=" * 60)
        logger.info("4DSG Summary:")
        logger.info("  Frames: %s", meta.get("num_frames"))
        logger.info("  Tracks: %s", meta.get("num_tracks"))
        logger.info("  Object crops: %d", len(object_crops))
        for t in tracks:
            oid = t["object_id"]
            fk = t["F_k"]
            extent = t.get("extent", [0, 0, 0])
            motion = t.get("motion", "stationary")
            img_pos = t.get("image_position", "?")
            logger.info(
                "  Object %d: %d obs, extent=[%.3f, %.3f, %.3f]m, %s, img=%s",
                oid, len(fk), extent[0], extent[1], extent[2], motion, img_pos,
            )
        logger.info("=" * 60)

    finally:
        try:
            sam3.end_all_runs()
        except Exception:
            pass
        # frame_dir no longer used (PIL images passed directly)

        timings["phase3_build_4dsg"] = time.time() - t0
        logger.info("TIMING phase3_build_4dsg: %.2fs", timings["phase3_build_4dsg"])

    logger.info("=" * 60)
    logger.info("TIMING SUMMARY:")
    total = 0.0
    for k, v in timings.items():
        logger.info("  %-35s %7.2fs", k, v)
        total += v
    logger.info("  %-35s %7.2fs", "TOTAL", total)
    logger.info("=" * 60)

    logger.info("All outputs in %s", OUT_DIR)
    logger.info("Done!")


if __name__ == "__main__":
    main()
