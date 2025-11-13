import argparse
import json
from pathlib import Path
import numpy as np
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="subtype_text_embeddings.npz", help="Input .npz file")
    p.add_argument("--out_centroids", "-o", default="text_centroids.npy", help="Output centroids .npy")
    p.add_argument("--out_subtypes", "-s", default="subtypes.json", help="Output subtypes JSON")
    p.add_argument("--force", action="store_true", help="Overwrite outputs if exist")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading {in_path} ...")
    data = np.load(str(in_path), allow_pickle=True)
    keys = list(data.files)
    if not keys:
        print("[ERROR] .npz contains no arrays", file=sys.stderr)
        sys.exit(2)

    centroid_keys = [k for k in keys if k.endswith("_centroid")]
    if not centroid_keys:
        centroid_keys = [k for k in keys if "centroid" in k.lower()]

    if not centroid_keys:
        print("[ERROR] Could not find centroid keys in the .npz. Available keys:", keys, file=sys.stderr)
        sys.exit(2)

    print("Detected centroid keys (in .npz order):")
    for k in centroid_keys:
        print("  -", k)

    subtypes = []
    centroids = []
    for k in centroid_keys:
        arr = data[k]
        if arr.ndim == 1:
            vec = arr.astype(np.float32)
        elif arr.ndim == 2 and 1 in arr.shape:
            vec = arr.reshape(-1).astype(np.float32)
        elif arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
            print(f"[WARN] Key {k} has shape {arr.shape}; averaging rows to form centroid.")
            vec = arr.mean(axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unrecognized shape for key {k}: {arr.shape}")

        if k.endswith("_centroid"):
            subtype = k[:-len("_centroid")]
        else:
            subtype = k
            for suf in ("_centroids", "centroids"):
                if subtype.endswith(suf):
                    subtype = subtype[:-len(suf)]
        subtype = subtype.strip()
        subtypes.append(subtype)
        centroids.append(vec)

    centroids = np.stack(centroids, axis=0)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    centroids_norm = centroids / norms

    out_cent = Path(args.out_centroids)
    out_sub = Path(args.out_subtypes)
    if (out_cent.exists() or out_sub.exists()) and not args.force:
        print(f"[ERROR] Output exists ({out_cent} or {out_sub}). Use --force to overwrite.", file=sys.stderr)
        sys.exit(2)

    np.save(str(out_cent), centroids_norm.astype(np.float32))
    with open(str(out_sub), "w") as f:
        json.dump(subtypes, f, indent=2)

    print(f"Saved centroids -> {out_cent} (shape: {centroids_norm.shape})")
    print(f"Saved subtype order -> {out_sub} (count: {len(subtypes)})")
    reloaded = np.load(str(out_cent))
    row_norms = np.linalg.norm(reloaded, axis=1)
    print("Row norms (min, mean, max):", float(row_norms.min()), float(row_norms.mean()), float(row_norms.max()))
    print("Subtypes (in order):", subtypes)

if __name__ == "__main__":
    main()
