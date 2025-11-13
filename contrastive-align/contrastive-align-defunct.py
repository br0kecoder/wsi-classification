import os
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#Note: did not work (14/10/25)

PATCH_EMB_DIR = r"D:\kidney_cells_classify\vit-encoder\patch_embeddings_per_slide"
TEXT_CENTROIDS_NPY = r"D:\kidney_cells_classify\text-encoder\text_centroids.npy"
SUBTYPES_JSON = r"D:\kidney_cells_classify\text-encoder\subtypes.json"
OUT_MODEL_DIR = "contrastive_model"
BATCH_SIZE = 8
MAX_TILES_PER_SLIDE = 512
EMBED_DIM_IMG = None
EMBED_DIM_TEXT = None
D_MODEL = 512
NUM_EPOCHS = 10
LR = 2e-4
TEMPERATURE = 0.07
TARGET_TEMPERATURE = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_MODEL_DIR, exist_ok=True)
print("Device:", DEVICE)

def load_text_centroids(centroid_path: str, subtypes_path: str) -> Tuple[np.ndarray, List[str]]:
    centroids = np.load(centroid_path)
    with open(subtypes_path) as f:
        subtypes = json.load(f)
    assert centroids.shape[0] == len(subtypes)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.clip(norms, 1e-12, None)
    return centroids.astype(np.float32), subtypes

class SlideEmbDataset(Dataset):
    def __init__(self, emb_dir: str, max_tiles: int = MAX_TILES_PER_SLIDE):
        self.emb_dir = Path(emb_dir)
        files = sorted(self.emb_dir.glob("*_embeddings.npy"))
        self.files = files
        self.max_tiles = max_tiles
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        n = arr.shape[0]
        if n > self.max_tiles:
            ids = np.random.choice(n, self.max_tiles, replace=False)
            arr = arr[ids]
        return {"slide_id": self.files[idx].stem.replace("_embeddings", ""), "embs": arr.astype(np.float32)}

def collate_fn(batch):
    max_len = max(item["embs"].shape[0] for item in batch)
    bs = len(batch)
    dim = batch[0]["embs"].shape[1]
    padded = np.zeros((bs, max_len, dim), dtype=np.float32)
    mask = np.zeros((bs, max_len), dtype=np.bool_)
    slide_ids = []
    for i, item in enumerate(batch):
        n = item["embs"].shape[0]
        padded[i, :n, :] = item["embs"]
        mask[i, :n] = 1
        slide_ids.append(item["slide_id"])
    return {"slide_ids": slide_ids, "embs": torch.from_numpy(padded), "mask": torch.from_numpy(mask)}

class ProjectionHeadQKV(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.d_model = d_model
        self.to_q = nn.Linear(in_dim, d_model, bias=False)
        self.to_k = nn.Linear(in_dim, d_model, bias=False)
        self.to_v = nn.Linear(in_dim, d_model, bias=False)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        if mask is not None:
            mask2 = mask.unsqueeze(1).expand(-1, attn_scores.size(1), -1)
            attn_scores = attn_scores.masked_fill(~mask2, float("-1e9"))
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)
        out = self.norm(out + self.ff(out))
        return out, attn

class AttentionMILpool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.att_linear = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1))
        self.norm = nn.LayerNorm(d_model)
    def forward(self, patch_feats, mask=None):
        scores = self.att_linear(patch_feats).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-1e9"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(patch_feats * weights.unsqueeze(-1), dim=1)
        return self.norm(pooled), weights

class SlideModel(nn.Module):
    def __init__(self, in_dim, text_dim, d_model=512):
        super().__init__()
        self.img_to_model = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.qkv = ProjectionHeadQKV(d_model, d_model)
        self.mil = AttentionMILpool(d_model)
        self.to_text = nn.Linear(d_model, text_dim)
        self.norm = nn.LayerNorm(text_dim)
    def forward(self, x, mask):
        x = self.img_to_model(x)
        x_ctx, attn_map = self.qkv(x, mask=mask)
        pooled, mil_weights = self.mil(x_ctx, mask=mask)
        z = self.to_text(pooled)
        z = self.norm(z)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
        return z, attn_map, mil_weights

def compute_initial_soft_targets(batch_embs: torch.Tensor, mask: torch.Tensor, text_centroids: torch.Tensor, temp=TARGET_TEMPERATURE):
    emb = batch_embs / (batch_embs.norm(dim=-1, keepdim=True) + 1e-12)
    sims = torch.einsum("bld,cd->blc", emb, text_centroids)
    sims = sims.masked_fill(~mask.unsqueeze(-1), float("-1e9"))
    sum_sims = sims.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)
    counts = mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
    mean_sims = sum_sims / counts
    soft = torch.softmax(mean_sims / temp, dim=-1)
    return soft

def train_epoch(model, dataloader, text_centroids_t, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="train"):
        embs = batch["embs"].to(device)
        mask = batch["mask"].to(device)
        with torch.no_grad():
            soft_targets = compute_initial_soft_targets(embs, mask, text_centroids_t, temp=TARGET_TEMPERATURE)
        z, attn_map, mil_weights = model(embs, mask)
        logits = (z @ text_centroids_t.T) / TEMPERATURE
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * embs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, text_centroids_t, device):
    model.eval()
    preds = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval"):
            embs = batch["embs"].to(device)
            mask = batch["mask"].to(device)
            z, attn_map, mil_weights = model(embs, mask)
            logits = (z @ text_centroids_t.T) / TEMPERATURE
            probs = F.softmax(logits, dim=-1)
            top1 = probs.argmax(dim=-1).cpu().numpy()
            for sid, ptop in zip(batch["slide_ids"], top1):
                preds[sid] = ptop
    return preds

def main():
    text_centroids_np, subtypes = load_text_centroids(TEXT_CENTROIDS_NPY, SUBTYPES_JSON)
    num_classes, text_dim = text_centroids_np.shape
    print("Loaded text centroids:", num_classes, "classes, dim", text_dim)
    ds = SlideEmbDataset(PATCH_EMB_DIR, max_tiles=MAX_TILES_PER_SLIDE)
    print("Found slides:", len(ds))
    if len(ds) == 0:
        raise RuntimeError("No slide embedding files found in " + PATCH_EMB_DIR)
    sample = np.load(sorted(Path(PATCH_EMB_DIR).glob("*_embeddings.npy"))[0])
    img_dim = sample.shape[1]
    print("Image embedding dim:", img_dim)
    model = SlideModel(in_dim=img_dim, text_dim=text_dim, d_model=D_MODEL).to(DEVICE)
    text_centroids_t = torch.from_numpy(text_centroids_np).to(DEVICE)
    text_centroids_t = text_centroids_t / (text_centroids_t.norm(dim=-1, keepdim=True) + 1e-12)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        train_loss = train_epoch(model, dl, text_centroids_t, optimizer, DEVICE)
        print(f"Epoch {epoch+1} train loss: {train_loss:.6f}")
        preds = evaluate(model, dl, text_centroids_t, DEVICE)
        counts = {}
        for v in preds.values():
            counts[v] = counts.get(v, 0) + 1
        print("Top-1 prototype counts:", {subtypes[k]: v for k, v in counts.items()})
    model_path = Path(OUT_MODEL_DIR) / "slide_model.pth"
    torch.save({"model_state": model.state_dict(), "config": {"D_MODEL": D_MODEL, "img_dim": img_dim, "text_dim": text_dim, "subtypes": subtypes}}, model_path)
    final_preds = evaluate(model, dl, text_centroids_t, DEVICE)
    preds_path = Path(OUT_MODEL_DIR) / "slide_top1_preds.json"
    final_named = {sid: subtypes[idx] for sid, idx in final_preds.items()}
    with open(preds_path, "w") as f:
        json.dump(final_named, f, indent=2)
    print("Saved model to", model_path)
    print("Saved final top1 predictions to", preds_path)

if __name__ == "__main__":
    main()
