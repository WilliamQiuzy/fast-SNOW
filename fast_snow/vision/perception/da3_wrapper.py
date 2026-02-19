"""DA3 (Depth Anything V3) wrapper for Fast-SNOW pipeline.

Implements Step 2: per-frame monocular depth estimation.
Output: depth map, intrinsics K, extrinsics T_wc, depth confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from fast_snow.engine.config.fast_snow_config import DA3Config

logger = logging.getLogger(__name__)


@dataclass
class DA3Result:
    """Output of DA3 inference for a single frame.

    Attributes:
        depth: (H, W) metric depth map in meters.
        K: (3, 3) camera intrinsics matrix.
        T_wc: (4, 4) world-to-camera transform (p_cam = T_wc @ p_world).
        depth_conf: (H, W) per-pixel depth confidence in [0, 1].
    """
    depth: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray
    depth_conf: np.ndarray

    @property
    def T_cw(self) -> np.ndarray:
        """Camera-to-world transform: p_world = T_cw @ p_cam."""
        return np.linalg.inv(self.T_wc)


class DA3Wrapper:
    """Wrapper around Depth Anything V3 for per-frame depth + pose estimation."""

    def __init__(self, config: Optional[DA3Config] = None):
        self.config = config or DA3Config()
        self._model = None

    def load(self) -> None:
        """Load the DA3 model. Called lazily on first inference."""
        if self._model is not None:
            return

        import sys
        da3_src = str(Path("fast_snow/vision/da3/src").resolve())
        if da3_src not in sys.path:
            sys.path.insert(0, da3_src)

        from depth_anything_3.api import DepthAnything3

        self._model = DepthAnything3.from_pretrained(self.config.model_path)
        self._model = self._model.to(self.config.device)
        self._model.eval()
        logger.info("DA3 model loaded from %s on %s", self.config.model_path, self.config.device)

    def infer(self, image: Union[np.ndarray, str, Path]) -> DA3Result:
        """Run DA3 on a single image.

        Args:
            image: RGB image as (H, W, 3) uint8 ndarray, or path to image file.

        Returns:
            DA3Result with depth, intrinsics, extrinsics, and confidence
            all at the **original** image resolution.
        """
        self.load()

        import torch
        from PIL import Image as PILImage

        # Remember original resolution for later resize
        if isinstance(image, (str, Path)):
            pil_img = PILImage.open(str(image)).convert("RGB")
            orig_h, orig_w = pil_img.height, pil_img.width
            image_input = [pil_img]
        elif isinstance(image, np.ndarray):
            orig_h, orig_w = image.shape[:2]
            pil_img = PILImage.fromarray(image)
            image_input = [pil_img]
        else:
            image_input = [image]
            orig_h = orig_w = None  # cannot resize without known dims

        with torch.inference_mode():
            prediction = self._model.inference(
                image=image_input,
                process_res=self.config.process_res,
                process_res_method=self.config.process_res_method,
            )

        # --- Validate required outputs (spec Step 2) ---
        if prediction.intrinsics is None:
            raise RuntimeError(
                "DA3 did not output intrinsics (K). "
                "Ensure model_path points to a model that supports camera estimation."
            )
        if prediction.extrinsics is None:
            raise RuntimeError(
                "DA3 did not output extrinsics (T_wc). "
                "Ensure model_path points to a model that supports pose estimation."
            )
        if not prediction.is_metric:
            raise RuntimeError(
                f"DA3 model at {self.config.model_path} outputs relative depth "
                f"(is_metric=0). Step 2 requires metric depth in metres. "
                f"Use a metric model (e.g. da3nested-giant-large)."
            )

        # Extract single-frame results (batch dim = 0)
        depth = prediction.depth[0]  # (proc_H, proc_W)
        K = prediction.intrinsics[0]
        T_wc = prediction.extrinsics[0]

        # Guarantee T_wc is (4, 4); DA3 may return (3, 4)
        if T_wc.ndim == 2 and T_wc.shape == (3, 4):
            T_wc_44 = np.eye(4, dtype=T_wc.dtype)
            T_wc_44[:3, :] = T_wc
            T_wc = T_wc_44
        elif T_wc.shape != (4, 4):
            raise RuntimeError(
                f"DA3 extrinsics has unexpected shape {T_wc.shape}; expected (4,4) or (3,4)."
            )
        if prediction.conf is not None:
            conf = prediction.conf[0].astype(np.float32)
        else:
            conf = np.ones_like(depth, dtype=np.float32)

        # --- Resize depth/conf to original resolution (align with SAM masks) ---
        #
        # DA3 forward pipeline:
        #   orig (orig_w, orig_h)
        #     → boundary resize → (resized_w, resized_h)    [K scaled]
        #     → (*crop only) center-crop to PATCH_SIZE multiples → (proc_w, proc_h)  [K cx/cy shifted]
        #
        # We invert in reverse order: undo crop, then undo resize.
        proc_h, proc_w = depth.shape
        if orig_h is not None and (proc_h != orig_h or proc_w != orig_w):
            import cv2

            method = self.config.process_res_method
            if "crop" in method:
                # 1) Recover the intermediate resize dimensions
                process_res = self.config.process_res
                if method.startswith("upper_bound"):
                    scale = process_res / max(orig_w, orig_h)
                else:  # lower_bound
                    scale = process_res / min(orig_w, orig_h)
                resized_w = max(1, round(orig_w * scale))
                resized_h = max(1, round(orig_h * scale))

                # 2) Undo center-crop (PATCH_SIZE alignment)
                crop_x = (resized_w - proc_w) // 2
                crop_y = (resized_h - proc_h) // 2
                K = K.copy()
                K[0, 2] += crop_x  # cx
                K[1, 2] += crop_y  # cy

                # depth/conf: pad back to (resized_h, resized_w)
                pad_bottom = resized_h - proc_h - crop_y
                pad_right = resized_w - proc_w - crop_x
                depth = cv2.copyMakeBorder(
                    depth, crop_y, pad_bottom, crop_x, pad_right,
                    cv2.BORDER_REPLICATE,
                )
                conf = cv2.copyMakeBorder(
                    conf, crop_y, pad_bottom, crop_x, pad_right,
                    cv2.BORDER_CONSTANT, value=0,
                )

                # 3) Undo boundary resize: scale K and depth/conf
                #    from (resized_h, resized_w) → (orig_h, orig_w)
                sx, sy = orig_w / resized_w, orig_h / resized_h
                K[0, :] *= sx
                K[1, :] *= sy
                depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                conf = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Resize methods: scale depth/conf and adjust K proportionally
                depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                conf = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                sx, sy = orig_w / proc_w, orig_h / proc_h
                K = K.copy()
                K[0, :] *= sx  # fx, s, cx
                K[1, :] *= sy  # fy, cy

        return DA3Result(
            depth=depth.astype(np.float32),
            K=K.astype(np.float64),
            T_wc=T_wc.astype(np.float64),
            depth_conf=conf,
        )
