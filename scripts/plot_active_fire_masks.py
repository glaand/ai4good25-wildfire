#!/usr/bin/env python3
"""
Quick utility to visualize active fire masks from a dataset (HDF5 per-fire files or folders of TIFFs).

Features:
- Load a single HDF5 file (expected dataset name: "data") or a folder of TIFFs
- Show RGB composite (user-selectable channels), raw active-fire channel, binary mask, temporal union of detections
- Overlay masks on RGB for quick inspection

Usage examples:
  python scripts/plot_active_fire_masks.py /path/to/fire.hdf5
  python scripts/plot_active_fire_masks.py /path/to/fire_folder --rgb 0,1,2 --active-ch -1

Notes:
- The code assumes the HDF5 dataset stores data in the shape (time, channels, height, width),
  and that the active-fire channel is the last channel by default (index -1). This matches
  the conventions used in `src/dataloader/FireSpreadDataset.py`.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import h5py
except Exception:
    h5py = None

try:
    import rasterio
except Exception:
    rasterio = None


def load_hdf5(path: str):
    if h5py is None:
        raise RuntimeError("h5py is required to read HDF5 files. Install via `pip install h5py`.")
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"HDF5 file {path} does not contain dataset 'data'. Found: {list(f.keys())}")
        data = f["data"][...]
        attrs = dict(f["data"].attrs)
    return data, attrs


def load_tif_stack_from_folder(folder: str):
    if rasterio is None:
        raise RuntimeError("rasterio is required to read TIFF files. Install via `pip install rasterio`.")
    tifs = sorted([str(p) for p in Path(folder).glob("*.tif")])
    if len(tifs) == 0:
        raise RuntimeError(f"No .tif files found in folder {folder}")
    imgs = []
    for t in tifs:
        with rasterio.open(t) as ds:
            # ds.read() -> (channels, H, W)
            imgs.append(ds.read())
    # stack into (time, channels, H, W)
    return np.stack(imgs, axis=0), {"files": tifs}


def channel_to_rgb(arr_channels: np.ndarray, rgb_indices):
    # arr_channels shape (channels, H, W)
    H = arr_channels.shape[1]
    W = arr_channels.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(3):
        idx = rgb_indices[i]
        if idx is None or idx >= arr_channels.shape[0] or idx < -arr_channels.shape[0]:
            rgb[:, :, i] = 0.0
        else:
            ch = arr_channels[idx]
            # normalize channel for display
            p1, p99 = np.nanpercentile(ch, (1, 99))
            if p99 - p1 <= 0:
                rgb[:, :, i] = np.clip(ch, 0, 1)
            else:
                rgb[:, :, i] = np.clip((ch - p1) / (p99 - p1), 0, 1)
    return rgb


def plot_masks(data: np.ndarray, attrs: dict, args):
    # data shape expected (time, channels, H, W)
    if data.ndim != 4:
        raise RuntimeError(f"Expected data with 4 dims (time, channels, H, W), got {data.shape}")

    time_idx = args.time_index if args.time_index is not None else data.shape[0] - 1
    time_idx = int(time_idx)
    time_idx = np.clip(time_idx, 0, data.shape[0] - 1)

    channels = data.shape[1]
    # resolve active channel index
    aidx = args.active_ch
    if aidx < 0:
        aidx = channels + aidx

    selected = data[time_idx]  # (channels, H, W)

    # compute masks
    raw_active = selected[aidx]
    binary_active = (raw_active > 0).astype(np.uint8)

    # temporal union (any detection across time)
    all_active = (data[:, aidx, :, :] > 0).any(axis=0).astype(np.uint8)
    sum_active = (data[:, aidx, :, :] > 0).sum(axis=0)

    # RGB
    rgb_idxs = [int(x) if x is not None else None for x in args.rgb]
    rgb = channel_to_rgb(selected, rgb_idxs)

    # plotting
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    axs[0].set_title(f"RGB (time={time_idx})")
    axs[0].imshow(rgb)
    axs[0].axis("off")

    axs[1].set_title("Raw active-fire channel (float)")
    im1 = axs[1].imshow(raw_active, cmap="magma")
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].set_title("Binary active-fire mask (time={})".format(time_idx))
    axs[2].imshow(binary_active, cmap="gray")
    axs[2].axis("off")

    axs[3].set_title("Temporal union of active detections")
    axs[3].imshow(all_active, cmap="gray")
    axs[3].axis("off")

    axs[4].set_title("Temporal sum of detections (counts)")
    im2 = axs[4].imshow(sum_active, cmap="viridis")
    axs[4].axis("off")
    fig.colorbar(im2, ax=axs[4], fraction=0.046, pad=0.04)

    # overlay union on RGB
    axs[5].set_title("RGB with union overlay")
    axs[5].imshow(rgb)
    # create red overlay where union is True
    mask = all_active.astype(bool)
    if mask.any():
        axs[5].contour(mask, colors="red", linewidths=0.5)
    axs[5].axis("off")

    plt.tight_layout()
    plt.savefig("active_fire_masks_plot.png")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Plot active fire masks from HDF5 or TIFF folders")
    p.add_argument("path", help="Path to .hdf5 file or folder with .tif files (one per time step)")
    p.add_argument("--time-index", type=int, help="Which time index to display (0-based). Defaults to last time step")
    p.add_argument("--active-ch", type=int, default=-1, help="Index of active fire channel (default: -1, last channel)")
    p.add_argument("--rgb", type=str, default="0,1,2", help="Comma-separated channel indices to use as R,G,B (default 0,1,2)")
    return p.parse_args()


def main():
    args = parse_args()
    # parse rgb indices
    rgb = [None, None, None]
    try:
        parts = [p.strip() for p in args.rgb.split(",")]
        for i in range(min(3, len(parts))):
            rgb[i] = int(parts[i]) if parts[i] != "" else None
    except Exception:
        print("Could not parse --rgb. Use e.g. --rgb 0,1,2")
        raise
    args.rgb = rgb

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")

    if path.is_file() and path.suffix in (".h5", ".hdf5"):
        data, attrs = load_hdf5(str(path))
    elif path.is_dir():
        data, attrs = load_tif_stack_from_folder(str(path))
    else:
        raise RuntimeError("Unsupported path type. Provide either a .hdf5 file or a folder with .tif files")

    plot_masks(data, attrs, args)


if __name__ == "__main__":
    main()
