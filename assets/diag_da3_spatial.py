#!/usr/bin/env python3
"""Check DA3 depth spatial structure after the fix."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

da3_src = str(Path("fast_snow/vision/da3/src").resolve())
if da3_src not in sys.path:
    sys.path.insert(0, da3_src)

import cv2
from PIL import Image as PILImage
from depth_anything_3.api import DepthAnything3

VIDEO = PROJECT_ROOT / "assets" / "examples_videos" / "horsing.mp4"
cap = cv2.VideoCapture(str(VIDEO))
ret, bgr = cap.read()
cap.release()
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
pil_img = PILImage.fromarray(rgb)

model = DepthAnything3.from_pretrained("fast_snow/models/da3")
model = model.to("cuda")
model.eval()

pred = model.inference(image=[pil_img], process_res=504, process_res_method="upper_bound_resize")

depth = pred.depth[0]  # (H, W)
H, W = depth.shape
print(f"Depth shape: {H}x{W}")
print(f"Overall range: [{depth.min():.4f}, {depth.max():.4f}]m")
print(f"Overall std: {depth.std():.4f}m")
print(f"Unique values: {len(np.unique(depth))}")

# Check spatial structure: row averages (top→bottom = far→near for typical scenes)
print(f"\n--- Row-average depth (top=row0, bottom=row{H-1}) ---")
for r in range(0, H, H // 10):
    row_mean = depth[r, :].mean()
    row_std = depth[r, :].std()
    print(f"  row {r:3d}: mean={row_mean:.4f}m, std={row_std:.4f}m")

# Check column structure
print(f"\n--- Column-average depth (left=col0, right=col{W-1}) ---")
for c in range(0, W, W // 8):
    col_mean = depth[:, c].mean()
    print(f"  col {c:3d}: mean={col_mean:.4f}m")

# Check quadrants
q1 = depth[:H//2, :W//2].mean()  # top-left
q2 = depth[:H//2, W//2:].mean()  # top-right
q3 = depth[H//2:, :W//2].mean()  # bottom-left
q4 = depth[H//2:, W//2:].mean()  # bottom-right
print(f"\n--- Quadrant means ---")
print(f"  top-left={q1:.4f}m, top-right={q2:.4f}m")
print(f"  bot-left={q3:.4f}m, bot-right={q4:.4f}m")

# Check gradient
top_mean = depth[:H//4, :].mean()
bot_mean = depth[3*H//4:, :].mean()
mid_mean = depth[H//4:3*H//4, :].mean()
print(f"\n--- Vertical bands ---")
print(f"  top quarter: {top_mean:.4f}m")
print(f"  middle half: {mid_mean:.4f}m")
print(f"  bottom quarter: {bot_mean:.4f}m")

# Check if center pixel differs from edges
center = depth[H//2, W//2]
corners = [depth[0,0], depth[0,-1], depth[-1,0], depth[-1,-1]]
print(f"\n--- Center vs corners ---")
print(f"  center: {center:.4f}m")
print(f"  corners: {[f'{c:.4f}' for c in corners]}")

# Also check the intrinsics
if pred.intrinsics is not None:
    K = pred.intrinsics[0]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    print(f"\n--- Intrinsics (resized to original) ---")
    print(f"  fx={fx:.1f}, fy={fy:.1f}")
    print(f"  cx={cx:.1f}, cy={cy:.1f}")
    # Compute horizontal FOV at median depth
    med_depth = np.median(depth)
    hfov_m = 2 * med_depth * np.tan(np.arctan(cx / fx))
    print(f"  Horizontal half-FOV: {np.degrees(np.arctan(cx/fx)):.1f}°")
    print(f"  At median depth {med_depth:.2f}m, horizontal extent: {hfov_m:.2f}m")

# Save depth as grayscale image for visual inspection
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
depth_vis = (depth_norm * 255).astype(np.uint8)
out_path = PROJECT_ROOT / "assets" / "depth_vis_horsing.png"
cv2.imwrite(str(out_path), depth_vis)
print(f"\nDepth visualization saved to: {out_path}")
print("(Bright = far, dark = near)")
