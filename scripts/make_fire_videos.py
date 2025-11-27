#!/usr/bin/env python3
"""
Generate consistent-color MP4 videos directly from active-fire masks
stored in per-fire HDF5 files (dataset: "data").

For each HDF5 file:
    - Load data: (time, channels, H, W)
    - Extract active fire channel (default: last channel)
    - Load real dates from HDF5 attribute "img_dates"
    - Build a video with:
          t = X h (YYYY-MM-DD)
    - Fixed color scale across frames
    - Binary or continuous mode

Usage:
    python make_fire_videos.py /path/to/hdf5_folder --out-dir videos
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
import imageio.v2 as imageio
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont


# -------------------------------------------------------------------------
# Create video frames with timestamp + date
# -------------------------------------------------------------------------
def make_video_from_arrays(arr_list, out_path: Path, img_dates,
                           fps=6, cmap_name="magma",
                           vmin=None, vmax=None,
                           timestep_hours=24.0, binary=False):
    """
    Create a video from raw 2D arrays using a fixed color scale,
    overlaying:  t = X h (YYYY-MM-DD)
    """

    # global min/max only for continuous values
    if not binary:
        if vmin is None:
            vmin = min(float(np.nanmin(a)) for a in arr_list)
        if vmax is None:
            vmax = max(float(np.nanmax(a)) for a in arr_list)
        cmap = cm.get_cmap(cmap_name)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264")

    # Load a readable font
    # We use a small size because fire tiles are small (128–512 px)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    for t, arr in enumerate(arr_list):

        # ---- Create RGB frame ----
        if binary:
            rgb = (arr.astype(np.uint8) * 255)
            rgb = np.stack([rgb] * 3, axis=-1)
        else:
            normed = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)
            rgb = (cmap(normed)[..., :3] * 255).astype(np.uint8)

        # ---- Convert to PIL for drawing text ----
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        # ---- Date extraction ----
        date_raw = img_dates[t]
        date_str = date_raw.decode("utf-8") if isinstance(date_raw, (bytes, np.bytes_)) else str(date_raw)

        # ---- Time formatting ----
        hours = t * timestep_hours
        text = f"t = {hours:.0f} h ({date_str})"

        # ---- Measure text size ----
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # ---- Safe placement ----
        margin = 5
        x = margin
        y = margin

        # ---- Background box ----
        draw.rectangle(
            [x - 2, y - 2, x + text_w + 2, y + text_h + 2],
            fill=(0, 0, 0, 120)   # semi-transparent black
        )

        # ---- Draw text ----
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

        writer.append_data(np.array(img))

    writer.close()
    print(f"[✓] Saved video with timestamps: {out_path}")


# -------------------------------------------------------------------------
# Process one fire HDF5 → MP4 video
# -------------------------------------------------------------------------
def process_single_hdf5(hdf5_path: Path, out_vid: Path,
                        active_ch: int = -1, binary=False, fps=6):

    print(f"[+] Processing {hdf5_path}")

    # Load full array + date strings
    with h5py.File(str(hdf5_path), "r") as f:
        data = f["data"][...]               # shape (T, C, H, W)
        img_dates = f["data"].attrs["img_dates"]

    T, C, H, W = data.shape

    # Resolve channel index
    ch = active_ch if active_ch >= 0 else C + active_ch

    # Extract raw frames
    frames = []
    for t in range(T):
        arr = data[t, ch]
        if binary:
            arr = (arr > 0).astype(np.uint8)
        frames.append(arr)

    # Build video (24 h per frame)
    make_video_from_arrays(
        frames,
        out_vid,
        img_dates=img_dates,
        fps=fps,
        binary=binary,
        timestep_hours=24.0
    )


# -------------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Create fire videos from HDF5 files with real dates & elapsed hours.")
    ap.add_argument("data_dir", help="Directory containing .h5/.hdf5 fire files")
    ap.add_argument("--out-dir", default="fire_videos", help="Where to save MP4s")
    ap.add_argument("--active-ch", type=int, default=-1, help="Active fire channel index (default: -1 = last)")
    ap.add_argument("--binary", action="store_true", default=True, help="Use binary masks instead of continuous values")
    ap.add_argument("--fps", type=int, default=6, help="Video framerate")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all fires
    files = sorted(list(data_dir.rglob("*.h5")) + list(data_dir.rglob("*.hdf5")))
    if len(files) == 0:
        print("No HDF5 files found.")
        return

    print(f"Found {len(files)} HDF5 fires.")

    # Process each fire into a video
    for f in files:
        out_vid = out_dir / f"{f.stem}.mp4"

        process_single_hdf5(
            f,
            out_vid,
            active_ch=args.active_ch,
            binary=args.binary,
            fps=args.fps
        )


if __name__ == "__main__":
    main()
