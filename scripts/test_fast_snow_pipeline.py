from __future__ import annotations

import numpy as np

from fast_snow.engine.config.fast_snow_config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_pipeline import (
    FastFrameInput,
    FastLocalDetection,
    FastSNOWPipeline,
)


def _cfg() -> FastSNOWConfig:
    cfg = FastSNOWConfig()
    cfg.step.grid_size = 4
    cfg.step.iou_threshold = 0.25
    cfg.depth_filter.min_points = 1
    cfg.depth_filter.max_extent = 1e6
    cfg.fusion.cross_run_iou_thresh = 0.1
    cfg.fusion.merge_centroid_dist_m = 10.0
    cfg.fusion.merge_temporal_gap = 10
    return cfg


def test_cross_run_fusion_merges_to_single_track():
    cfg = _cfg()
    pipe = FastSNOWPipeline(cfg)

    depth = np.full((8, 8), 1.2, dtype=np.float32)
    K = np.eye(3, dtype=np.float64)
    T_wc = np.eye(4, dtype=np.float64)

    m1 = np.zeros((8, 8), dtype=bool)
    m2 = np.zeros((8, 8), dtype=bool)
    m1[2:6, 2:6] = True
    m2[3:7, 2:6] = True  # overlaps heavily with m1

    frame = FastFrameInput(
        frame_idx=0,
        depth_t=depth,
        K_t=K,
        T_wc_t=T_wc,
        detections=[
            FastLocalDetection(run_id="car", local_obj_id=1, mask=m1, score=0.95),
            FastLocalDetection(run_id="automobile", local_obj_id=7, mask=m2, score=0.80),
        ],
    )

    pipe.process_frame(frame)
    out = pipe.build_4dsg_dict()

    assert out["metadata"]["num_tracks"] == 1
    assert len(out["tracks"]) == 1


def test_no_truncation_and_no_rounding_in_step_json():
    cfg = _cfg()
    pipe = FastSNOWPipeline(cfg)

    K = np.eye(3, dtype=np.float64)
    T_wc = np.eye(4, dtype=np.float64)

    for t in range(3):
        depth = np.full((8, 8), 1.123456 + 0.05 * t, dtype=np.float64)
        depth[3, 3] = 1.654321 + 0.03 * t

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True

        pipe.process_frame(
            FastFrameInput(
                frame_idx=t,
                depth_t=depth,
                K_t=K,
                T_wc_t=T_wc,
                detections=[FastLocalDetection(run_id="car", local_obj_id=1, mask=mask, score=0.9)],
            )
        )

    out = pipe.build_4dsg_dict()
    assert out["metadata"]["num_tracks"] == 1

    track = out["tracks"][0]
    assert len(track["F_k"]) == 3  # no observation truncation

    first = track["F_k"][0]
    # theta should be track-level span
    assert first["theta"] == [0, 2]

    z = first["c"][2]
    assert abs(z - round(z, 2)) > 1e-6
