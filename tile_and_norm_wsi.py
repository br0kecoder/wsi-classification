import os
import csv
import argparse
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
import numpy as np
from tqdm import tqdm
import traceback
import math
import pandas as pd

def norm_HnE(img, Io=240, alpha=1, beta=0.15): #formula from paper
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img_reshaped = img.reshape((-1, 3)).astype(np.float64)

    OD = -np.log10((img_reshaped + 1) / Io)

    ODhat = OD[~np.any(OD < beta, axis=1)]


    if ODhat.shape[0] < 3:
        # not enough tissue pixels to estimate
        Inorm = img.astype(np.uint8)
        H = np.zeros_like(Inorm)
        E = np.zeros_like(Inorm)
        return (Inorm, H, E)

    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T
    C, *_ = np.linalg.lstsq(HE, Y, rcond=None)

    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef, out=np.ones_like(maxC), where=maxCRef != 0)
    C2 = np.divide(C, tmp[:, np.newaxis])

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    return (Inorm, H, E)


def is_empty_patch(tile_np, bright_threshold=230, empty_fraction_threshold=0.5):

    if tile_np.ndim != 3 or tile_np.shape[2] < 3:
        return True, 1.0 # bad data, removing 
    bright_mask = np.all(tile_np[:, :, :3] > bright_threshold, axis=2)
    bright_count = int(np.sum(bright_mask))
    total_pixels = tile_np.shape[0] * tile_np.shape[1]
    fraction = bright_count / float(total_pixels)
    return fraction > empty_fraction_threshold, fraction


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def process_slide(slide_path, out_root, tile_size=256, overlap=0, level_index=None,
                  bright_threshold=230, empty_frac_thresh=0.5,
                  max_tiles_per_slide=None, save_h_and_e=False):
    
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    print(f"Processing slide: {slide_name}")
    meta = []

    try:
        slide = open_slide(slide_path)
    except Exception as e:
        print(f"Could not open {slide_path}: {e}")
        print(traceback.format_exc())
        return meta

    tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=False)

    if level_index is None:
        level_index = tiles.level_count - 1

    cols, rows = tiles.level_tiles[level_index]

    out_slide_dir = os.path.join(out_root, slide_name)
    out_tiles_dir = os.path.join(out_slide_dir, "tiles")
    ensure_dir(out_tiles_dir)

    total_saved = 0
    for row in range(rows):
        for col in range(cols):
            try:
                tile = tiles.get_tile(level_index, (col, row))
                tile = tile.convert("RGB")
                tile_np = np.array(tile)

                is_empty, bright_frac = is_empty_patch(tile_np, bright_threshold, empty_frac_thresh)
                if is_empty:
                    
                    meta.append({
                        "slide": slide_name,
                        "tile_name": f"{col}_{row}.png",
                        "col": col, "row": row,
                        "tile_w": tile_np.shape[1],
                        "tile_h": tile_np.shape[0],
                        "bright_frac": bright_frac,
                        "kept": False,
                        "path": ""
                    })
                    continue


                Inorm, H, E = norm_HnE(tile_np, Io=240, alpha=1, beta=0.15)

                tile_filename = f"{col}_{row}.png"
                out_tile_path = os.path.join(out_tiles_dir, tile_filename)
                Image.fromarray(Inorm).save(out_tile_path)

                if save_h_and_e:
                    Image.fromarray(H).save(os.path.join(out_tiles_dir, f"{col}_{row}_H.png"))
                    Image.fromarray(E).save(os.path.join(out_tiles_dir, f"{col}_{row}_E.png"))

                meta.append({
                    "slide": slide_name,
                    "tile_name": tile_filename,
                    "col": col, "row": row,
                    "tile_w": tile_np.shape[1],
                    "tile_h": tile_np.shape[0],
                    "bright_frac": bright_frac,
                    "kept": True,
                    "path": os.path.relpath(out_tile_path)
                })

                total_saved += 1
                if max_tiles_per_slide is not None and total_saved >= max_tiles_per_slide:
                    print(f"Reached max_tiles_per_slide={max_tiles_per_slide} for {slide_name}")
                    break

            except Exception as e:
                print(f"Failed tile col={col}, row={row} on {slide_name}: {e}")
                print(traceback.format_exc())
        if max_tiles_per_slide is not None and total_saved >= max_tiles_per_slide:
            break

    print(f"Saved {total_saved} tiles for slide {slide_name}")
    return meta


def find_slide_files(dirs, exts=(".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs", ".png", ".jpg", ".jpeg")):
    
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"Not a directory: {d}")
            continue
        for root, _, filenames in os.walk(d):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    files.append(os.path.join(root, fn))
    return sorted(files)


def main():
    DIR1 = "data/DHMC_wsi_01"
    DIR2 = "data/DHMC_wsi_02"
    OUT_ROOT = "images/saved_tiles/normalized_tiles"
    TILE_SIZE = 256
    OVERLAP = 0
    BRIGHT_THRESHOLD = 230
    EMPTY_FRAC_THRESH = 0.5
    MAX_TILES_PER_SLIDE = None
    SAVE_HE = False # false for now, not needing both h&e

    slide_dirs = [DIR1, DIR2]
    slides = find_slide_files(slide_dirs)
    if len(slides) == 0:
        print("No slide files found in the provided directories. Exiting.")
        return

    print(f"Found {len(slides)} slide files.")

    ensure_dir(OUT_ROOT)

    all_meta = []
    for slide_path in tqdm(slides, desc="slides"):
        meta = process_slide(
            slide_path,
            out_root=OUT_ROOT,
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            level_index=None,
            bright_threshold=BRIGHT_THRESHOLD,
            empty_frac_thresh=EMPTY_FRAC_THRESH,
            max_tiles_per_slide=MAX_TILES_PER_SLIDE,
            save_h_and_e=SAVE_HE
        )
        all_meta.extend(meta)

    meta_df = pd.DataFrame(all_meta)
    out_csv = os.path.join(OUT_ROOT, "tiles_metadata.csv")
    meta_df.to_csv(out_csv, index=False)
    print(f"Tile metadata written to {out_csv}")

if __name__ == "__main__":
    main()