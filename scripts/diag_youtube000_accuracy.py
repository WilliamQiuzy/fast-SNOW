"""Accuracy diagnostic for youtube000.mp4 (skateboard scene).

Verifies:
  1. Camera pose: plots 3D camera trajectory, checks smoothness
  2. 3D centroid accuracy: reprojects world centroids back to 2D and overlays on frames
  3. Trajectory continuity: checks for jumps between consecutive frames

Outputs:
  assets/diag_youtube000/
    sam3/              — SAM3 per-frame mask overlays
    centroid_overlay/  — frames with reprojected 3D centroids
    camera_trajectory.png  — 3D camera path
    trajectory_analysis.png — per-object centroid smoothness
    4dsg.json          — 4DSG scene graph
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

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
logger = logging.getLogger("diag_yt000")

VIDEO = ROOT / "benchmark" / "VLM4D-video" / "videos_real" / "youtube-vos" / "youtube000.mp4"
OUT_DIR = ROOT / "assets" / "diag_youtube000"

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (0, 128, 255), (128, 0, 255), (200, 200, 200), (100, 100, 100),
]


def draw_masks_on_frame(rgb, masks, labels=None, alpha=0.45):
    vis = rgb.copy()
    for i, m in enumerate(masks):
        mask_bool = m if m.dtype == bool else m > 0.5
        color = COLORS[i % len(COLORS)]
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)
        if labels and i < len(labels):
            ys, xs = np.where(mask_bool)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                cv2.putText(vis, str(labels[i]), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def extract_frames(video_path, target_fps):
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


def save_frames_as_jpeg(frames, out_dir):
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


def reproject_world_to_pixel(point_3d, T_wc, K):
    """Reproject a 3D world point to 2D pixel coordinates.

    Args:
        point_3d: (3,) world coordinate [x, y, z]
        T_wc: (4,4) world-to-camera transform
        K: (3,3) camera intrinsics

    Returns:
        (u, v) pixel coordinates, or None if behind camera
    """
    p_h = np.array([*point_3d, 1.0], dtype=np.float64)
    p_cam = T_wc @ p_h  # camera coordinates
    if p_cam[2] <= 0:
        return None  # behind camera
    p_img = K @ p_cam[:3]
    u = float(p_img[0] / p_img[2])
    v = float(p_img[1] / p_img[2])
    return (u, v)


def plot_camera_trajectory(da3_results, timestamps, out_path):
    """Plot 3D camera trajectory and save as image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    positions = []
    for r in da3_results:
        T_cw = np.linalg.inv(r.T_wc)
        pos = T_cw[:3, 3]
        positions.append(pos)
    positions = np.array(positions)

    fig = plt.figure(figsize=(16, 6))

    # 3D trajectory
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b.-", markersize=3)
    ax1.scatter(*positions[0], color="green", s=100, label="start", zorder=5)
    ax1.scatter(*positions[-1], color="red", s=100, label="end", zorder=5)
    ax1.set_xlabel("X (right)")
    ax1.set_ylabel("Y (down)")
    ax1.set_zlabel("Z (forward)")
    ax1.set_title("Camera 3D trajectory")
    ax1.legend()

    # XZ trajectory (top-down view)
    ax2 = fig.add_subplot(132)
    ax2.plot(positions[:, 0], positions[:, 2], "b.-", markersize=3)
    ax2.scatter(positions[0, 0], positions[0, 2], color="green", s=100, label="start", zorder=5)
    ax2.scatter(positions[-1, 0], positions[-1, 2], color="red", s=100, label="end", zorder=5)
    ax2.set_xlabel("X (right)")
    ax2.set_ylabel("Z (forward)")
    ax2.set_title("Top-down (XZ)")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True)

    # Per-axis over time
    ax3 = fig.add_subplot(133)
    ts = np.array(timestamps)
    ax3.plot(ts, positions[:, 0], "r-", label="X (right)")
    ax3.plot(ts, positions[:, 1], "g-", label="Y (down)")
    ax3.plot(ts, positions[:, 2], "b-", label="Z (forward)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Position (m)")
    ax3.set_title("Camera position over time")
    ax3.legend()
    ax3.grid(True)

    # Smoothness: compute inter-frame velocity
    fig2_text = []
    speeds = []
    for i in range(1, len(positions)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt > 0:
            v = np.linalg.norm(positions[i] - positions[i - 1]) / dt
            speeds.append(v)
    if speeds:
        fig2_text.append(f"Speed: mean={np.mean(speeds):.3f}m/s, max={np.max(speeds):.3f}m/s, std={np.std(speeds):.3f}m/s")
        fig2_text.append(f"Total displacement: {np.linalg.norm(positions[-1] - positions[0]):.3f}m")
    fig.suptitle("\n".join(fig2_text), fontsize=10, y=0.02)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Camera trajectory saved to %s", out_path)

    return positions, speeds


def plot_object_trajectories(pipeline, da3_results, timestamps, out_path):
    """Plot per-object 3D centroid trajectories and smoothness analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = pipeline._tracks
    n_tracks = len([t for t in tracks.values() if len(t.observations) >= 2])
    if n_tracks == 0:
        logger.warning("No tracks with >=2 observations for trajectory analysis")
        return

    fig, axes = plt.subplots(min(n_tracks, 10), 3, figsize=(18, 3 * min(n_tracks, 10)))
    if min(n_tracks, 10) == 1:
        axes = axes[np.newaxis, :]

    plot_idx = 0
    for gid in sorted(tracks.keys()):
        if plot_idx >= 10:
            break
        state = tracks[gid]
        if len(state.observations) < 2:
            continue

        obs = sorted(state.observations, key=lambda x: x.frame_idx)
        ts = [o.timestamp_s for o in obs]
        centroids = np.array([[o.step.centroid.x, o.step.centroid.y, o.step.centroid.z] for o in obs])

        # Per-axis position
        axes[plot_idx, 0].plot(ts, centroids[:, 0], "r.-", label="X")
        axes[plot_idx, 0].plot(ts, centroids[:, 1], "g.-", label="Y")
        axes[plot_idx, 0].plot(ts, centroids[:, 2], "b.-", label="Z")
        axes[plot_idx, 0].set_title(f"Object {gid}: XYZ over time")
        axes[plot_idx, 0].legend(fontsize=7)
        axes[plot_idx, 0].grid(True)

        # Inter-frame displacement (smoothness check)
        disps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        dts = np.diff(ts)
        speeds_obj = disps / np.maximum(dts, 1e-6)
        axes[plot_idx, 1].plot(ts[1:], disps, "k.-")
        axes[plot_idx, 1].set_title(f"Object {gid}: inter-frame disp (m)")
        axes[plot_idx, 1].grid(True)
        # Flag jumps
        if len(disps) > 1:
            median_d = np.median(disps)
            jumps = disps > max(3 * median_d, 0.5)
            if jumps.any():
                jump_ts = np.array(ts[1:])[jumps]
                jump_ds = disps[jumps]
                axes[plot_idx, 1].scatter(jump_ts, jump_ds, color="red", s=50, zorder=5, label="jump!")
                axes[plot_idx, 1].legend(fontsize=7)

        # Speed histogram
        axes[plot_idx, 2].hist(speeds_obj, bins=15, color="steelblue", edgecolor="black")
        axes[plot_idx, 2].set_title(f"Object {gid}: speed distribution (m/s)")
        axes[plot_idx, 2].axvline(np.median(speeds_obj), color="red", linestyle="--", label=f"median={np.median(speeds_obj):.2f}")
        axes[plot_idx, 2].legend(fontsize=7)

        plot_idx += 1

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Object trajectory analysis saved to %s", out_path)


def main():
    import torch

    timings = {}

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for d in ["sam3", "centroid_overlay"]:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

    config = ROSEConfig(
        sampling=SamplingConfig(target_fps=10.0),
        da3=DA3Config(chunk_size=0, chunk_overlap=5),
        fastsam=FastSAMConfig(conf_threshold=0.55, iou_threshold=0.9),
        sam3=SAM3Config(offload_state_to_cpu=True, offload_video_to_cpu=True,
                        chunk_size=-1),
    )

    # ── Step 0: Extract frames ──
    t0 = time.time()
    frames, src_indices, timestamps = extract_frames(VIDEO, 10.0)
    n = len(frames)
    frame_dir = save_frames_as_jpeg(frames, Path(tempfile.mkdtemp(prefix="rose_yt000_")))
    timings["step0_extract"] = time.time() - t0
    logger.info("Extracted %d frames, saved to %s", n, frame_dir)

    try:
        # ── Phase 1: DA3 ──
        logger.info("=== Phase 1: DA3 batch inference ===")
        t0 = time.time()
        da3 = DA3Wrapper(config.da3)
        da3_results = da3.infer_batch(frames)
        da3.unload()
        torch.cuda.empty_cache()
        timings["phase1_da3"] = time.time() - t0
        logger.info("DA3 done: %d results in %.1fs", len(da3_results), timings["phase1_da3"])

        # ── Camera pose analysis ──
        logger.info("=== Camera Pose Analysis ===")
        cam_positions, cam_speeds = plot_camera_trajectory(da3_results, timestamps, OUT_DIR / "camera_trajectory.png")
        logger.info("Camera positions (first 5):")
        for i in range(min(5, len(cam_positions))):
            p = cam_positions[i]
            logger.info("  t=%.2fs: [%.4f, %.4f, %.4f]", timestamps[i], p[0], p[1], p[2])
        logger.info("Camera positions (last 5):")
        for i in range(max(0, len(cam_positions) - 5), len(cam_positions)):
            p = cam_positions[i]
            logger.info("  t=%.2fs: [%.4f, %.4f, %.4f]", timestamps[i], p[0], p[1], p[2])
        if cam_speeds:
            logger.info("Camera speed: mean=%.3fm/s, max=%.3fm/s, std=%.3fm/s",
                        np.mean(cam_speeds), np.max(cam_speeds), np.std(cam_speeds))
        total_disp = np.linalg.norm(cam_positions[-1] - cam_positions[0])
        logger.info("Camera total displacement: %.3fm", total_disp)
        logger.info("Camera is_metric: %s", da3_results[0].is_metric)

        # ── Phase 2a: FastSAM + SAM3 init ──
        t0 = time.time()
        logger.info("=== Phase 2a: FastSAM frame 0 + SAM3 init ===")
        fastsam = FastSAMWrapper(config.fastsam)
        sam3 = SAM3SharedSessionManager(config.sam3)
        sam3.set_video_dir(frame_dir)

        fastsam_per_frame = {}
        dets_0 = fastsam.detect(frames[0])
        fastsam_per_frame[0] = dets_0
        logger.info("FastSAM frame 0: %d detections", len(dets_0))

        bboxes = [list(d.bbox_xywh_norm) for d in dets_0]
        if bboxes:
            _, init_masks = sam3.create_run_with_initial_bboxes(
                boxes_xywh=bboxes, box_labels=[1] * len(bboxes),
                frame_idx=0, tag="fastsam_bbox",
            )

            # ── SAM3 acceleration settings ──
            sam3._predictor.model.retain_feature_cache = True
            sam3._predictor.model.detector.retain_multigpu_buffer = True
            sam3._predictor.model.tracker.num_maskmem = 4
            sam3._predictor.model.tracker.memory_temporal_stride_for_eval = 2
            sam3._predictor.model.partial_propagation_stride = 2
            sam3._predictor.model.full_propagation_stride = 2
            sam3._predictor.model.vg_detection_stride = 2
            logger.info("SAM3 acceleration: cache_retention + num_maskmem=4 + mem_stride=2 + prop_stride=2 + full_stride=2 + vg_det_stride=2")

            fastsam.unload()
            torch.cuda.empty_cache()

            logger.info("SAM3 propagation over %d frames...", n)
            for fidx in range(n):
                sam3.propagate_all(fidx)

            if len(bboxes) > 1:
                for i in range(1, len(bboxes)):
                    bx, by, bw, bh = bboxes[i]
                    sam3.add_object_point(0, (bx + bw / 2.0, by + bh / 2.0))
        else:
            fastsam.unload()

        timings["phase2a"] = time.time() - t0

        # ── Phase 2b: Discovery ──
        t0 = time.time()
        logger.info("=== Phase 2b: Discovery ===")
        discovery_thresh = config.fastsam.discovery_iou_thresh
        max_disc = config.sam3.max_discovery_per_frame
        min_mask_frac = config.sam3.discovery_min_mask_frac
        max_total = config.sam3.max_discovery_total
        discovery_stride = config.sam3.full_propagation_stride
        new_obj_count = 0

        # Cap discovery budget based on max_active_tracks
        max_active = config.sam3.max_active_tracks
        if max_active > 0:
            current_active = len(sam3._obj_id_to_run_id)
            active_headroom = max(0, max_active - current_active)
            max_total = min(max_total, active_headroom) if max_total > 0 else active_headroom
            logger.info("Discovery cap: %d active objects, headroom %d", current_active, active_headroom)

        if sam3.active_runs:
            for fidx in range(1, n):
                if discovery_stride > 1 and fidx % discovery_stride != 0:
                    continue
                if max_total > 0 and new_obj_count >= max_total:
                    break
                dets = fastsam.detect(frames[fidx])
                # FastSAM already returns detections sorted by mask area
                # descending — larger objects are more likely foreground.
                fastsam_per_frame[fidx] = dets
                cached = sam3.propagate_all(fidx)
                dh, dw = frames[fidx].shape[:2]
                frame_disc = 0
                for det in dets:
                    if max_disc > 0 and frame_disc >= max_disc:
                        break
                    if max_total > 0 and new_obj_count >= max_total:
                        break
                    if min_mask_frac > 0 and det.mask.sum() < min_mask_frac * dh * dw:
                        continue
                    if not any(mask_iou(det.mask, m.mask) >= discovery_thresh for m in cached):
                        cy_px, cx_px = mask_centroid(det.mask)
                        h, w = det.mask.shape[:2]
                        sam3.add_object_point(fidx, (cx_px / w, cy_px / h))
                        new_obj_count += 1
                        frame_disc += 1

        logger.info("Discovery: %d new objects", new_obj_count)
        fastsam.unload()
        torch.cuda.empty_cache()
        timings["phase2b_discovery"] = time.time() - t0

        # ── Phase 2c: Partial propagation ──
        t0 = time.time()
        frame0_pt_count = max(0, len(bboxes) - 1) if bboxes else 0
        total_new = frame0_pt_count + new_obj_count
        if total_new > 0:
            logger.info("=== Phase 2c: Partial propagation (%d new objects) ===", total_new)
            sam3.propagate_new_objects()
        timings["phase2c"] = time.time() - t0

        # ── Save SAM3 visualizations ──
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

        # ── Phase 3: Build 4DSG + accuracy analysis ──
        logger.info("=== Phase 3: Building 4DSG ===")
        t0 = time.time()
        pipeline = ROSEPipeline(config)
        crop_pad = config.vlm.object_crop_padding
        crop_sz = config.vlm.object_crop_size
        best_crops = {}

        # Store per-frame per-object data for centroid overlay
        # frame_centroids[fidx] = [(gid, centroid_xyz, mask_center_2d), ...]
        frame_centroids = {}

        for fidx in range(n):
            sam3_masks = list(sam3.propagate_all(fidx))
            best = {}
            for m in sam3_masks:
                key = (m.run_id, m.obj_id_local)
                if key not in best or m.score > best[key].score:
                    best[key] = m
            sam3_masks = list(best.values())

            for m in sam3_masks:
                key = (m.run_id, m.obj_id_local)
                crop = _crop_object_from_mask(frames[fidx], m.mask, padding=crop_pad, size=crop_sz)
                brightness = float(crop.mean())
                prev = best_crops.get(key)
                if prev is None or (m.score, brightness) > (prev[1], prev[3]):
                    best_crops[key] = (crop, m.score, src_indices[fidx], brightness)

            detections = [
                FastLocalDetection(run_id=m.run_id, local_obj_id=m.obj_id_local,
                                   mask=m.mask, score=m.score)
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
            pipeline.process_frame(fi)

            # Collect centroid info for this frame
            fc = []
            for m in sam3_masks:
                key = (m.run_id, m.obj_id_local)
                gid = pipeline._local_to_global.get(key)
                if gid is not None and gid in pipeline._tracks:
                    state = pipeline._tracks[gid]
                    # Get the latest observation for this frame
                    for obs in reversed(state.observations):
                        if obs.frame_idx == src_indices[fidx]:
                            c = obs.step.centroid
                            fc.append((gid, np.array([c.x, c.y, c.z]), obs.mask_center_2d))
                            break
            frame_centroids[fidx] = fc

        # Save crops
        crops_dir = OUT_DIR / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        object_crops = {}
        for key, gid in pipeline._local_to_global.items():
            if key in best_crops and gid not in object_crops:
                crop_rgb, _score, src_idx, _bright = best_crops[key]
                if crop_rgb.mean() < 1.0:
                    continue
                crop_path = crops_dir / f"obj_{gid:04d}.jpg"
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                object_crops[gid] = {"path": str(crop_path), "source_frame_idx": src_idx}

        # Dedup
        n_before = len(object_crops)
        object_crops = pipeline.merge_duplicate_tracks(object_crops)
        n_after = len(object_crops)
        if n_before != n_after:
            logger.info("Dedup: %d → %d tracks", n_before, n_after)

        dsg = pipeline.build_4dsg_dict(object_crops=object_crops)
        dsg_json = json.dumps(dsg, indent=2, ensure_ascii=False, sort_keys=False)
        (OUT_DIR / "4dsg.json").write_text(dsg_json)
        logger.info("4DSG saved")

        timings["phase3_build"] = time.time() - t0

        # ── Centroid reprojection overlay ──
        logger.info("=== Centroid Reprojection Overlay ===")
        reproject_errors = []  # (gid, fidx, pixel_error_from_mask_center)

        for fidx in range(n):
            vis = frames[fidx].copy()
            h, w = vis.shape[:2]
            T_wc = da3_results[fidx].T_wc
            K = da3_results[fidx].K

            for gid, centroid_3d, mask_center_2d in frame_centroids.get(fidx, []):
                # Skip objects that got deduped away
                if gid not in object_crops and gid not in pipeline._tracks:
                    continue

                uv = reproject_world_to_pixel(centroid_3d, T_wc, K)
                if uv is None:
                    continue
                u, v = int(round(uv[0])), int(round(uv[1]))

                # Mask center in pixels
                mc_u = int(round(mask_center_2d[0] * w))
                mc_v = int(round(mask_center_2d[1] * h))

                # Pixel error between reprojected centroid and mask center
                err = np.sqrt((uv[0] - mc_u) ** 2 + (uv[1] - mc_v) ** 2)
                reproject_errors.append((gid, fidx, err))

                color = COLORS[gid % len(COLORS)]

                # Draw reprojected 3D centroid as cross
                if 0 <= u < w and 0 <= v < h:
                    cv2.drawMarker(vis, (u, v), color, cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(vis, f"o{gid}", (u + 10, v - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw mask center as circle
                if 0 <= mc_u < w and 0 <= mc_v < h:
                    cv2.circle(vis, (mc_u, mc_v), 6, color, 2)

                # Draw line between them
                if 0 <= u < w and 0 <= v < h and 0 <= mc_u < w and 0 <= mc_v < h:
                    cv2.line(vis, (u, v), (mc_u, mc_v), color, 1)

            # Add frame info
            cv2.putText(vis, f"Frame {fidx} t={timestamps[fidx]:.2f}s", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "X=3D centroid reproj, O=mask center", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(OUT_DIR / "centroid_overlay" / f"frame_{fidx:04d}.jpg"), bgr)

        # Reprojection error analysis
        if reproject_errors:
            errs = np.array([e[2] for e in reproject_errors])
            logger.info("=== Reprojection Error Analysis ===")
            logger.info("  Total measurements: %d", len(errs))
            logger.info("  Mean pixel error: %.1f px", np.mean(errs))
            logger.info("  Median pixel error: %.1f px", np.median(errs))
            logger.info("  Max pixel error: %.1f px", np.max(errs))
            logger.info("  Std pixel error: %.1f px", np.std(errs))
            logger.info("  <10px: %d (%.1f%%)", (errs < 10).sum(), (errs < 10).sum() / len(errs) * 100)
            logger.info("  <50px: %d (%.1f%%)", (errs < 50).sum(), (errs < 50).sum() / len(errs) * 100)
            logger.info("  <100px: %d (%.1f%%)", (errs < 100).sum(), (errs < 100).sum() / len(errs) * 100)

            # Per-object error
            per_obj = {}
            for gid, fidx, err in reproject_errors:
                per_obj.setdefault(gid, []).append(err)
            for gid in sorted(per_obj.keys()):
                obj_errs = per_obj[gid]
                logger.info("  Object %d: n=%d, mean=%.1fpx, max=%.1fpx",
                            gid, len(obj_errs), np.mean(obj_errs), np.max(obj_errs))

        # ── Object trajectory analysis ──
        logger.info("=== Object Trajectory Analysis ===")
        plot_object_trajectories(pipeline, da3_results, timestamps, OUT_DIR / "trajectory_analysis.png")

        # Log per-track summary
        for gid in sorted(pipeline._tracks.keys()):
            if gid not in object_crops:
                continue
            state = pipeline._tracks[gid]
            obs = sorted(state.observations, key=lambda x: x.frame_idx)
            if len(obs) < 2:
                continue
            centroids = np.array([[o.step.centroid.x, o.step.centroid.y, o.step.centroid.z] for o in obs])
            disps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
            total_disp = np.linalg.norm(centroids[-1] - centroids[0])
            median_step = np.median(disps)
            max_step = np.max(disps)
            jumps = (disps > max(3 * median_step, 0.5)).sum()
            logger.info("  Object %d: %d obs, total_disp=%.3fm, median_step=%.4fm, max_step=%.4fm, jumps=%d",
                        gid, len(obs), total_disp, median_step, max_step, jumps)
            logger.info("    First centroid: [%.3f, %.3f, %.3f]", *centroids[0])
            logger.info("    Last  centroid: [%.3f, %.3f, %.3f]", *centroids[-1])

        # ── 4DSG Summary ──
        meta = dsg.get("metadata", {})
        tracks = dsg.get("tracks", [])
        logger.info("=" * 60)
        logger.info("4DSG Summary:")
        logger.info("  Frames: %s", meta.get("num_frames"))
        logger.info("  Tracks: %s", meta.get("num_tracks"))
        logger.info("  Camera motion: %s", meta.get("camera_motion"))
        for t in tracks:
            oid = t["object_id"]
            fk = t["F_k"]
            theta = t.get("theta", [0, 0])
            extent = t.get("extent", [0, 0, 0])
            motion = t.get("motion", "stationary")
            img_pos = t.get("image_position", "?")
            logger.info("  Object %d: %d obs, theta=[%.3f, %.3f]s, extent=[%.3f, %.3f, %.3f]m, %s, img=%s",
                        oid, len(fk), theta[0], theta[1], extent[0], extent[1], extent[2], motion, img_pos)
        logger.info("=" * 60)

    finally:
        try:
            sam3.end_all_runs()
        except Exception:
            pass
        shutil.rmtree(frame_dir, ignore_errors=True)

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
