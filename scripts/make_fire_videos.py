#!/usr/bin/env python3
"""
Generate consistent-color MP4 videos directly from active-fire masks
stored in per-fire HDF5 files (dataset: "data").

For each HDF5 file:
    - Load data with shape (time, channels, H, W)
    - Extract the active fire channel (default: -1)
    - Build a video with fixed color scaling across all frames
    - No matplotlib plots -> no autoscaling, no flicker

Usage:
    python make_fire_videos.py /path/to/hdf5_dir --out-dir videos
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
import imageio.v2 as imageio
from matplotlib import cm


# ------------------------------------------------------------
# Convert a list of 2D arrays into a consistent-color video
# ------------------------------------------------------------
def make_video_from_arrays(arr_list, out_path: Path, fps=6,
                           cmap_name="magma", vmin=None, vmax=None):
    """
    Create a video from raw 2D numpy arrays using a fixed color scale.
    """
    # Determine global min/max if not provided
    if vmin is None:
        vmin = min(float(np.nanmin(a)) for a in arr_list)
    if vmax is None:
        vmax = max(float(np.nanmax(a)) for a in arr_list)

    cmap = cm.get_cmap(cmap_name)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264")

    for arr in arr_list:
        # normalize fixed scale
        normed = (arr - vmin) / (vmax - vmin + 1e-12)
        normed = np.clip(normed, 0, 1)

        # convert to RGB using colormap
        rgb = (cmap(normed)[..., :3] * 255).astype(np.uint8)
        writer.append_data(rgb)

    writer.close()
    print(f"[✓] Saved video: {out_path}")


# ------------------------------------------------------------
# Process a single HDF5 file into a video
# ------------------------------------------------------------
def process_single_hdf5(hdf5_path: Path, out_vid: Path,
                        active_ch: int = -1, binary=False, fps=6):
    """
    Convert an HDF5 fire file into a video.
    """
    print(f"[+] Processing {hdf5_path}")

    # Load data
    with h5py.File(str(hdf5_path), "r") as f:
        if "data" not in f:
            raise RuntimeError(f"File missing 'data' dataset: {hdf5_path}")
        data = f["data"][...]     # shape (T, C, H, W)

    T, C, H, W = data.shape

    # Resolve channel index
    ch = active_ch if active_ch >= 0 else C + active_ch
    if ch < 0 or ch >= C:
        raise RuntimeError(f"Invalid active channel index: {active_ch}")

    # Extract frames as raw arrays
    frames = []
    for t in range(T):
        arr = data[t, ch]  # 2D
        if binary:
            arr = (arr > 0).astype(np.uint8)
        frames.append(arr)

    # Make video
    make_video_from_arrays(frames, out_vid, fps=fps,
                           cmap_name="magma",
                           vmin=None, vmax=None)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Make consistent-color fire videos from HDF5 files")
    ap.add_argument("data_dir", help="directory containing .h5/.hdf5 fire files")
    ap.add_argument("--out-dir", default="fire_videos", help="directory to save MP4 videos")
    ap.add_argument("--active-ch", type=int, default=-1, help="active fire channel index (default: -1)")
    ap.add_argument("--binary", action="store_true", help="use binary mask instead of raw values")
    ap.add_argument("--fps", type=int, default=6, help="video frame rate")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    files = sorted(list(data_dir.rglob("*.h5")) + list(data_dir.rglob("*.hdf5")))
    if len(files) == 0:
        print("No HDF5 files found.")
        return

    print(f"Found {len(files)} HDF5 files.")

    for f in files:
        out_vid = out_dir / f"{f.stem}.mp4"
        process_single_hdf5(f, out_vid,
                            active_ch=args.active_ch,
                            binary=args.binary,
                            fps=args.fps)


if __name__ == "__main__":
    main()
