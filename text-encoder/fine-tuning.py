
import torch
from torch import nn, optim
from transformers import CLIPModel, CLIPTokenizer
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PromptDataset(Dataset):
    def __init__(self, prompts, labels=None):
        self.prompts = prompts
        self.labels = labels
    def __len__(self): return len(self.prompts)
    def __getitem__(self, i):
        return self.prompts[i], (self.labels[i] if self.labels is not None else -1)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
for p in model.parameters():
    p.requires_grad = False

n_classes = 4
classifier = nn.Sequential(
    nn.Linear(model.text_projection.shape[0], 256),
    nn.ReLU(),
    nn.Linear(256, n_classes)
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-5)

prompts = ["A histopathology slide..."] * 100
labels = [0] * 100  
dataset = PromptDataset(prompts, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model.eval()
for epoch in range(5):
    classifier.train()
    for prompt_batch, label_batch in loader:
        tokenized = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_feats = model.get_text_features(**tokenized)  
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        logits = classifier(text_feats)  
        loss = criterion(logits, torch.tensor(label_batch, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss {loss.item():.4f}")

for p in model.parameters(): p.requires_grad = False
model.save_pretrained("./clip_text_finetuned_and_frozen")
tokenizer.save_pretrained("./clip_text_finetuned_and_frozen")
torch.save(classifier.state_dict(), "./text_classifier_head.pt")
