import os
import random
from glob import glob
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

CLASS_NAMES=["Benign", "Chromophobe", "Clearcell", "Oncocytoma", "Papillary"]
CLASS_TO_IDX={c: i for i, c in enumerate(CLASS_NAMES)}

METADATA_CSV="metadata.csv"
TILES_ROOT="images/saved_tiles/normalized_tiles"
OUT_DIR="models_resnet50" #model set to resnet50 for now
TRAIN_FRAC=0.7
VAL_FRAC=0.15
TEST_FRAC=0.15
EPOCHS=12
BATCH_SIZE=64
LR=1e-4
WORKERS=4
SEED=42

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_metadata(metadata_csv):
    df=pd.read_csv(metadata_csv)
    if not {"File Name", "Diagnosis"}.issubset(df.columns):
        raise ValueError("metadata.csv must contain 'File Name' and 'Diagnosis' columns")
    mapping={}
    for _, r in df.iterrows():
        fn=str(r["File Name"]).strip()
        slide=os.path.splitext(os.path.basename(fn))[0]
        mapping[slide]=str(r["Diagnosis"]).strip()
    return mapping

def gather_tiles_per_slide(tiles_root, slide_to_label):
    slides_info={}
    slide_dirs=[d for d in glob(os.path.join(tiles_root, "*")) if os.path.isdir(d)]
    for sd in sorted(slide_dirs):
        slide_name=os.path.basename(sd)
        tiles_dir=os.path.join(sd, "tiles")
        if not os.path.isdir(tiles_dir):
            continue
        all_tiles=[]
        for ext in("*.png", "*.jpg", "*.jpeg"):
            all_tiles.extend(glob(os.path.join(tiles_dir, ext)))
        tiles=[p for p in sorted(all_tiles) if not (p.lower().endswith("_h.png") or p.lower().endswith("_e.png"))]
        if not tiles:
            continue
        label=slide_to_label.get(slide_name, None)
        slides_info[slide_name]={"label": label, "tiles": tiles}
    return slides_info

class TileDataset(Dataset):
    def __init__(self, samples, transform=None, return_slide=False):
        self.samples=samples
        self.transform=transform
        self.return_slide=return_slide

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, slide=self.samples[idx]
        im=Image.open(path).convert("RGB")
        if self.transform:
            im_t=self.transform(im)
        else:
            im_t=transforms.ToTensor()(im)
        if self.return_slide:
            return im_t, label, slide, path
        return im_t, label

def build_model(num_classes=len(CLASS_NAMES), pretrained=True):
    model=models.resnet50(pretrained=pretrained)
    in_feats=model.fc.in_features
    model.fc=nn.Linear(in_feats, num_classes)
    return model

def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)

def predict_single_image(model_path, image_path, device=None):
    
    if device is None:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chk=load_checkpoint(model_path, device)
    class_names=chk.get("class_names", CLASS_NAMES)
    model=build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(chk["model_state_dict"])
    model=model.to(device)
    model.eval()

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    im=Image.open(image_path).convert("RGB")
    x=transform(im).unsqueeze(0).to(device)
    with torch.no_grad():
        logits=model(x)
        probs=torch.softmax(logits, dim=1).cpu().numpy()[0]
    class_idx=int(probs.argmax())
    prob_map={class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return class_names[class_idx], prob_map

def evaluate_tilelevel(model, dataloader, device):
    model.eval()
    preds=[]
    labels=[]
    probs=[]
    paths=[]
    slides=[]
    soft=nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in dataloader:
            if len(batch)==4:
                images, lbls, slide_names, paths_batch=batch
            else:
                images, lbls=batch
                slide_names=None
                paths_batch=[None]*len(lbls)
            images=images.to(device)
            logits=model(images)
            p=soft(logits).cpu().numpy()
            preds.extend(p.argmax(axis=1).tolist())
            probs.extend(p.tolist())
            labels.extend(lbls.numpy().tolist())
            paths.extend(paths_batch)
            if slide_names is not None:
                slides.extend(slide_names)
            else:
                slides.extend([None]*len(lbls))
    return preds, labels, probs, slides, paths

def aggregate_slide_predictions(probs_list, slides_list, true_labels_for_slide):
    slide_accum={}
    for probs, slide in zip(probs_list, slides_list):
        if slide not in slide_accum:
            slide_accum[slide]=[]
        slide_accum[slide].append(np.array(probs))
    slide_preds={}
    slide_probs={}
    for slide, arrs in slide_accum.items():
        avg=np.mean(np.stack(arrs, axis=0), axis=0)
        slide_probs[slide]=avg
        slide_preds[slide]=int(np.argmax(avg))
    slides=list(slide_preds.keys())
    y_true=[true_labels_for_slide[s] for s in slides]
    y_pred=[slide_preds[s] for s in slides]
    return slides, y_true, y_pred, slide_probs

def train_singleprocess(train_samples, val_samples, test_samples, device):
    print("Starting single-process training on device:", device)

    TILE_SIZE=256

    transform_train=transforms.Compose([
        transforms.Resize((TILE_SIZE, TILE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    transform_eval=transforms.Compose([
        transforms.Resize((TILE_SIZE, TILE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds=TileDataset(train_samples, transform=transform_train)
    val_ds=TileDataset(val_samples, transform=transform_eval, return_slide=True)
    test_ds=TileDataset(test_samples, transform=transform_eval, return_slide=True)

    train_loader=DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    val_loader=DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
    test_loader=DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    model=build_model(len(CLASS_NAMES), pretrained=True).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=LR)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=True)

    best_val_acc=0.0
    best_path=os.path.join(OUT_DIR, "resnet50_best.pth")

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss=0.0
        correct=0
        total=0
        for imgs, labels in tqdm(train_loader, desc=f"Train E{epoch}"):
            imgs=imgs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            logits=model(imgs)
            loss=criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()*imgs.size(0)
            preds=logits.argmax(dim=1)
            correct+=(preds==labels).sum().item()
            total+=imgs.size(0)
        train_loss=running_loss/total if total else 0.0
        train_acc=correct/total if total else 0.0
        print(f"Epoch {epoch} - train loss {train_loss:.4f}, acc {train_acc:.4f}")

        preds_val, labels_val, probs_val, slides_val, _=evaluate_tilelevel(model, val_loader, device)
        val_acc=accuracy_score(labels_val, preds_val) if labels_val else 0.0
        print(f"  Val tile-acc: {val_acc:.4f} (samples {len(labels_val)})")
        scheduler.step(val_acc)

        if val_acc>best_val_acc:
            best_val_acc=val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": CLASS_NAMES
            }, best_path)
            print(f"  Saved best model to {best_path}")

    print("Training finished. Best val acc:", best_val_acc)

    chk=load_checkpoint(best_path, device)
    model.load_state_dict(chk["model_state_dict"])
    model=model.to(device)
    preds_test, labels_test, probs_test, slides_test, paths_test=evaluate_tilelevel(model, test_loader, device)

    print("\nTile-level Test accuracy:", accuracy_score(labels_test, preds_test))
    print("Tile-level classification report:")
    print(classification_report(labels_test, preds_test, target_names=CLASS_NAMES, digits=4))

    true_labels_slide={}
    for _, lab, slide in test_samples:
        if slide not in true_labels_slide:
            true_labels_slide[slide]=lab
    slides_list, y_true_slide, y_pred_slide, slide_probs=aggregate_slide_predictions(probs_test, slides_test, true_labels_slide)
    print("\nSlide-level Test accuracy:", accuracy_score(y_true_slide, y_pred_slide))
    print("Slide-level classification report:")
    print(classification_report(y_true_slide, y_pred_slide, target_names=CLASS_NAMES, digits=4))

    out_pred_csv=os.path.join(OUT_DIR, "test_tile_predictions.csv")
    rows=[]
    for p, l, prob, slide, path in zip(preds_test, labels_test, probs_test, slides_test, paths_test):
        rows.append({"path": path, "true": CLASS_NAMES[l], "pred": CLASS_NAMES[p], "slide": slide})
    pd.DataFrame(rows).to_csv(out_pred_csv, index=False)
    print("Saved tile predictions to", out_pred_csv)

    return best_path

def main():    
    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    slide_to_label=read_metadata(METADATA_CSV)
    print(f"Metadata entries: {len(slide_to_label)}")
    slides_info=gather_tiles_per_slide(TILES_ROOT, slide_to_label)
    if not slides_info:
        raise RuntimeError(f"No slides with tiles found under {TILES_ROOT}")
    label_tile_counts=Counter()
    total_tiles=0
    for slide, info in slides_info.items():
        label=info["label"] if info["label"] is not None else "Unknown"
        n=len(info["tiles"])
        label_tile_counts[label]+=n
        total_tiles+=n
    print("Tile counts per label:")
    for label, cnt in label_tile_counts.items():
        print(f"  {label}: {cnt} tiles")
    print("Total tiles:", total_tiles)

    valid_slides=[s for s, info in slides_info.items() if info["label"] in CLASS_TO_IDX]
    if not valid_slides:
        raise RuntimeError("No slides found with labels matching CLASS_NAMES.")
    valid_labels=[slides_info[s]["label"] for s in valid_slides]

    slides_train, slides_temp, y_train, y_temp=train_test_split(
        valid_slides, valid_labels, test_size=(1-TRAIN_FRAC), stratify=valid_labels, random_state=SEED
    )
    relative_val=VAL_FRAC/(VAL_FRAC+TEST_FRAC)
    slides_val, slides_test, y_val, y_test=train_test_split(
        slides_temp, y_temp, test_size=(1-relative_val), stratify=y_temp, random_state=SEED
    )
    print(f"Split slides: train={len(slides_train)}, val={len(slides_val)}, test={len(slides_test)}")

    def build_samples(slide_list):
        s=[]
        for slide in slide_list:
            lab=slides_info[slide]["label"]
            idx=CLASS_TO_IDX[lab]
            for p in slides_info[slide]["tiles"]:
                s.append((p, idx, slide))
        return s

    train_samples=build_samples(slides_train)
    val_samples=build_samples(slides_val)
    test_samples=build_samples(slides_test)
    print(f"Tile counts by split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path=train_singleprocess(train_samples, val_samples, test_samples, device)
    print("Best model path:", best_path)
if __name__ == "__main__":
    main()