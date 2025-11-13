import torch
import numpy as np
from transformers import CLIPModel, CLIPTokenizer
from transformers import AutoProcessor, AutoModel

MODEL_NAME = "openai/clip-vit-base-patch32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#MODEL_NAME = AutoModel.from_pretrained("google/medsiglip-448").to(device)
#processor = AutoProcessor.from_pretrained("google/medsiglip-448")
OUT_EMB_FILE = "subtype_text_embeddings.npz"
SAVED_MODEL_DIR = "./frozen_clip_text_encoder"

subtype_prompts = {
    "clear_cell": [
        "A histopathology slide depicting clear cell renal cell carcinoma (ccRCC), characterized by clear cytoplasm and distinct cell borders.",
        "Microscopic image of kidney tissue showing clear cell renal cell carcinoma with clear cytoplasm, prominent nucleoli, and vascular structures.",
        "Clear cell RCC, a subtype of renal cell carcinoma, showing abnormally large clear cells in the renal cortex with high-grade nuclear features.",
        "Renal biopsy revealing clear cell carcinoma in the kidney, showing well-defined cytoplasmic clearing and a rich blood supply in the tumor.",
        "Clear cell carcinoma of the kidney under a microscope, featuring cells with clear cytoplasm, round nuclei, and a well-defined cell membrane."
    ],
    "papillary": [
        "A histopathology slide showing papillary renal cell carcinoma (pRCC), featuring papillary structures with branching fibrovascular cores.",
        "Microscopic image of kidney tissue with papillary renal cell carcinoma, demonstrating papillary projections with delicate vascular cores and abundant foamy cytoplasm.",
        "Papillary RCC, a subtype of renal cell carcinoma, showing irregular papillae with cystic spaces, focal necrosis, and calcifications.",
        "A renal biopsy with papillary RCC, demonstrating papillary architectures with thin-walled blood vessels and prominent nucleoli.",
        "Histological examination of kidney tissue showing papillary renal carcinoma with branching papillae, and cells with eosinophilic cytoplasm."
    ],
    "chromophobe": [
        "Histopathology slide of chromophobe renal cell carcinoma, characterized by pale cytoplasm, prominent cell borders, and perinuclear clearing.",
        "Kidney tissue showing chromophobe RCC, featuring cells with pale eosinophilic cytoplasm and distinctive, irregular nuclei.",
        "Chromophobe renal carcinoma, featuring cells with a characteristic pale cytoplasm and irregular nuclei with prominent nucleoli.",
        "Renal biopsy showing chromophobe RCC, with cells exhibiting a uniform cytoplasmic pale appearance and abundant mitochondria.",
        "Microscopic image of kidney tissue demonstrating chromophobe RCC, with cells having granular cytoplasm and perinuclear halos."
    ],
    "benign": [
        "Normal kidney tissue showing healthy glomeruli, proximal tubules, and distal tubules, with no signs of malignancy.",
        "Renal tissue biopsy revealing benign kidney cells with normal architecture and absence of neoplastic growth.",
        "Histopathology slide showing normal renal parenchyma, with healthy nephron structures and no evidence of tumor or dysplasia.",
        "Microscopic examination of normal kidney tissue, with intact renal structures including glomeruli, tubules, and interstitial tissue.",
        "Benign renal tissue showing well-preserved glomeruli and tubular structures with no evidence of carcinoma or cystic changes."
    ]
}

tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

all_embeddings = {}
model.eval()
with torch.no_grad():
    for subtype, prompts in subtype_prompts.items():
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        text_feats = model.get_text_features(**tokenized)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.cpu().numpy()
        all_embeddings[subtype] = text_feats

centroids = {}
for subtype, embs in all_embeddings.items():
    centroid = np.mean(embs, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    centroids[subtype] = centroid

np.savez(OUT_EMB_FILE, **{f"{k}_variants": v for k, v in all_embeddings.items()},
         **{f"{k}_centroid": v for k, v in centroids.items()})
print(f"Saved embeddings & centroids to {OUT_EMB_FILE}")

for p in model.text_model.parameters():
    p.requires_grad = False
if hasattr(model, "text_projection"):
    model.text_projection.requires_grad = False

model.save_pretrained(SAVED_MODEL_DIR)
tokenizer.save_pretrained(SAVED_MODEL_DIR)
print(f"Saved frozen model + tokenizer to {SAVED_MODEL_DIR}")
