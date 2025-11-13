import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import clip

TILES_ROOT = r"D:\kidney_cells_classify\dataset-kidney-dartmouth\images\saved_tiles\normalized_tiles"
OUT_DIR = "patch_embeddings_per_slide"
MANIFEST = os.path.join(OUT_DIR, "manifest.json")
MODEL_NAME = "ViT-B/32"
BATCH_SIZE = 64
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading CLIP model:", MODEL_NAME, "on", DEVICE)
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False

def list_tile_files(slide_tiles_dir):
    p = Path(slide_tiles_dir)
    if not p.exists():
        return []
    files = [x for x in sorted(p.iterdir()) if x.suffix.lower() in IMG_EXTS and x.is_file()]
    return files

def process_slide(slide_dir, out_dir, batch_size=BATCH_SIZE):
    slide_id = os.path.basename(slide_dir.rstrip("/\\"))
    tiles_dir = os.path.join(slide_dir, "tiles")
    tile_paths = list_tile_files(tiles_dir)
    if not tile_paths:
        return None

    out_file = os.path.join(out_dir, f"{slide_id}_embeddings.npy")
    filenames_file = os.path.join(out_dir, f"{slide_id}_filenames.json")

    if os.path.exists(out_file) and os.path.exists(filenames_file):
        arr = np.load(out_file, mmap_mode="r")
        with open(filenames_file, "r") as f:
            filenames = json.load(f)
        return {
            "slide_id": slide_id,
            "emb_file": os.path.relpath(out_file),
            "n_tiles": arr.shape[0],
            "embedding_dim": arr.shape[1],
            "filenames": filenames
        }

    all_embs = []
    idx = 0
    pbar = tqdm(range(0, len(tile_paths), batch_size), desc=f"Slide {slide_id}", unit="batch")
    for start in pbar:
        batch_paths = tile_paths[start:start+batch_size]
        images = []
        for tp in batch_paths:
            try:
                img = Image.open(tp).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                print(f"Warning: failed to load {tp}: {e}. Skipping.")
                continue
        if not images:
            continue
        batch_tensor = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            emb = model.encode_image(batch_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy()
            all_embs.append(emb)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if not all_embs:
        return None

    emb_array = np.vstack(all_embs)
    np.save(out_file, emb_array)
    filenames = [str(p) for p in tile_paths]
    with open(filenames_file, "w") as f:
        json.dump(filenames, f, indent=2)

    return {
        "slide_id": slide_id,
        "emb_file": os.path.relpath(out_file),
        "n_tiles": emb_array.shape[0],
        "embedding_dim": emb_array.shape[1],
        "filenames": filenames_file
    }

def main():
    manifest = {}
    slides = sorted([os.path.join(TILES_ROOT, d) for d in os.listdir(TILES_ROOT)
                     if os.path.isdir(os.path.join(TILES_ROOT, d))])
    print(f"Found {len(slides)} slide directories in {TILES_ROOT}")

    for slide_path in slides:
        meta = process_slide(slide_path, OUT_DIR)
        if meta:
            manifest[meta["slide_id"]] = meta
        else:
            print(f"No tiles processed for {slide_path}")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved manifest to", MANIFEST)
    total_tiles = sum(v["n_tiles"] for v in manifest.values())
    print(f"Total slides processed: {len(manifest)}, total tiles: {total_tiles}")

if __name__ == "__main__":
    main()