"""Inspect a 4DSG with an arbitrary video path.  Same as inspect_4dsg.py but
takes the full video path instead of just a stem."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/workspace/fast-SNOW")


def main():
    video = Path(sys.argv[1]).resolve()
    stem = video.stem
    fdsg_path = ROOT / "benchmark" / "fdsg_one" / f"{stem}.4dsg.json"
    out_jpg = ROOT / "benchmark" / "fdsg_one" / f"{stem}_inspect.jpg"
    out_json = ROOT / "benchmark" / "fdsg_one" / f"{stem}_tracks.json"

    fdsg = json.loads(fdsg_path.read_text())
    tracks = fdsg["tracks"]

    cap = cv2.VideoCapture(str(video))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_idx = np.linspace(0, nf - 1, 8).astype(int)
    keyframes = []
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            keyframes.append((int(idx), frame))
    cap.release()

    kf_strip_h = 144
    kf_strip_w = 256
    kf_imgs = []
    for idx, frame in keyframes:
        rs = cv2.resize(frame, (kf_strip_w, kf_strip_h))
        t_s = idx / fps if fps > 0 else 0
        cv2.putText(rs, f"f{idx} t={t_s:.2f}s", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        kf_imgs.append(rs)
    kf_row = np.hstack(kf_imgs)

    crop_sz = 200
    n_cols = 6
    n_rows = (len(tracks) + n_cols - 1) // n_cols if tracks else 1
    grid = np.full((n_rows * (crop_sz + 24), n_cols * crop_sz, 3), 240, dtype=np.uint8)

    track_summary = []
    for i, tr in enumerate(tracks):
        r, c = divmod(i, n_cols)
        y0, x0 = r * (crop_sz + 24), c * crop_sz
        crop_path = Path(tr["visual_anchor"]["path"])
        if not crop_path.exists():
            cv2.rectangle(grid, (x0+1, y0+1), (x0+crop_sz-1, y0+crop_sz-1), (0,0,255), 2)
            mean_b = -1
        else:
            crop = cv2.imread(str(crop_path))
            if crop is None:
                crop = np.zeros((crop_sz, crop_sz, 3), dtype=np.uint8)
            crop = cv2.resize(crop, (crop_sz, crop_sz))
            mean_b = float(crop.mean())
            grid[y0:y0+crop_sz, x0:x0+crop_sz] = crop

        oid = tr["object_id"]
        n_obs = len(tr["F_k"])
        ext = tr["extent"]
        pos = tr["image_position"]
        fk = tr["F_k"]
        t0, t1 = fk[0]["t"], fk[-1]["t"]
        cap_y = y0 + crop_sz + 14
        cv2.putText(grid, f"obj{oid} n={n_obs} {pos} mean={mean_b:.0f}",
                    (x0+4, cap_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        cv2.putText(grid, f"ext={ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f}",
                    (x0+4, cap_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (40,40,40), 1)
        track_summary.append({
            "object_id": oid, "n_obs": n_obs, "t_range": [t0, t1],
            "extent": ext, "image_position": pos,
            "crop_path": str(crop_path),
        })

    if kf_row.shape[1] < grid.shape[1]:
        pad = np.full((kf_strip_h, grid.shape[1] - kf_row.shape[1], 3), 200, dtype=np.uint8)
        kf_row = np.hstack([kf_row, pad])
    elif kf_row.shape[1] > grid.shape[1]:
        kf_row = kf_row[:, :grid.shape[1]]

    title = np.full((36, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(title, f"{stem} | {len(tracks)} tracks | {fdsg['metadata']['num_frames']} sampled frames",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
    final = np.vstack([title, kf_row, grid])
    cv2.imwrite(str(out_jpg), final, [cv2.IMWRITE_JPEG_QUALITY, 88])
    out_json.write_text(json.dumps(track_summary, indent=2))
    print(f"Wrote {out_jpg}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
