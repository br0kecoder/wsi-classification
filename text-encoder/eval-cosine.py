import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return x / norms

def evaluate_text_centroids(
    centroids_npy: str = "text_centroids.npy",
    subtypes_json: str = "subtypes.json",
    variants_npz: Optional[str] = None
) -> Dict[str, Any]:
    cent_path = Path(centroids_npy)
    if not cent_path.exists():
        raise FileNotFoundError(f"Centroids file not found: {centroids_npy}")
    centroids = np.load(str(cent_path)).astype(np.float32)
    if centroids.ndim != 2:
        raise ValueError("text_centroids.npy must be a 2D array (C, D).")
    C, D = centroids.shape

    sub_path = Path(subtypes_json)
    if not sub_path.exists():
        raise FileNotFoundError(f"Subtypes JSON not found: {subtypes_json}")
    with open(sub_path, "r") as f:
        subtypes = json.load(f)
    if len(subtypes) != C:
        raise ValueError(f"Number of subtypes ({len(subtypes)}) does not match rows in centroids ({C}).")

    centroids = l2_normalize_rows(centroids)

    sim_mat = centroids @ centroids.T 

    results: Dict[str, Any] = {
        "similarity_matrix": sim_mat,
        "centroids": centroids,
        "subtypes": subtypes,
        "accuracy": None,
        "per_class_accuracy": None,
        "confusion_matrix": None,
        "intra_class_mean": None,
        "inter_class_mean": None
    }
    inter_means = []
    for i in range(C):
        others = np.delete(sim_mat[i], i)
        inter_means.append(float(np.mean(others)))
    results["inter_class_mean"] = inter_means
    if variants_npz is not None:
        var_path = Path(variants_npz)
        if not var_path.exists():
            raise FileNotFoundError(f"Variants .npz not found: {variants_npz}")
        data = np.load(str(var_path), allow_pickle=True)
        keys = list(data.files)
        variant_keys = {}
        for k in keys:
            if k.endswith("_variants"):
                base = k[:-len("_variants")]
                variant_keys[base] = k
        if not variant_keys:
            for k in keys:
                if "variant" in k.lower():
                    variant_keys[k] = k

        confusion = np.zeros((C, C), dtype=int)
        per_class_correct = np.zeros(C, dtype=int)
        per_class_total = np.zeros(C, dtype=int)
        intra_means = []

        for i, subtype in enumerate(subtypes):
            vkey = variant_keys.get(subtype)
            if vkey is None:
                match = None
                for base, k in variant_keys.items():
                    if base.lower() == subtype.lower():
                        match = k
                        break
                vkey = match
            if vkey is None:
                intra_means.append(float("nan"))
                per_class_total[i] = 0
                per_class_correct[i] = 0
                continue

            arr = data[vkey].astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
            
            sims = arr @ centroids.T
            preds = np.argmax(sims, axis=1)
            per_class_total[i] = sims.shape[0]
            per_class_correct[i] = int((preds == i).sum())
            for p in preds:
                confusion[i, int(p)] += 1
            
            intra_means.append(float(sims[:, i].mean()) if sims.shape[0] > 0 else float("nan"))

        total = int(per_class_total.sum())
        correct = int(per_class_correct.sum())
        overall_acc = correct / total if total > 0 else None
        per_class_acc = {
            subtypes[i]: (int(per_class_correct[i]), int(per_class_total[i]),
                          (per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else None))
            for i in range(C)
        }

        results["accuracy"] = overall_acc
        results["per_class_accuracy"] = per_class_acc
        results["confusion_matrix"] = confusion
        results["intra_class_mean"] = intra_means

    return results

if __name__ == "__main__":
    res = evaluate_text_centroids(
        centroids_npy="text_centroids.npy",
        subtypes_json="subtypes.json",
        variants_npz="subtype_text_embeddings.npz"  
    )
    print("Subtypes (order):", res["subtypes"])
    print("Centroid similarity matrix:\n", np.round(res["similarity_matrix"], 3))
    print("Inter-class mean similarities:", np.round(res["inter_class_mean"], 3))
    if res["accuracy"] is not None:
        print("Overall variant->centroid accuracy: {:.4f}".format(res["accuracy"]))
        print("Per-class accuracy (correct, total, acc):")
        for k, v in res["per_class_accuracy"].items():
            print("  ", k, v)
        print("Confusion matrix (rows=true, cols=pred):\n", res["confusion_matrix"])
        print("Intra-class mean sims:", np.round(res["intra_class_mean"], 3))
    else:
        print("No variants provided — only centroid similarity stats computed.")
