import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset
from transforms import train_transform, val_transform
from model import load_efficientnet_b0
from utils import accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# Dataset
# ------------------
train_ds = build_dataset("data/train", train_transform)
val_ds   = build_dataset("data/val", val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

# ------------------
# Model
# ------------------
model = load_efficientnet_b0(
    num_classes=10,
    weight_path="weights/efficientnet_b0_base.pt"
).to(DEVICE)

# 🔒 1단계: classifier만 학습
for p in model.features.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# ------------------
# Training
# ------------------
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ------------------
    # Validation
    # ------------------
    model.eval()
    val_acc = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_acc += accuracy(outputs, labels)

    val_acc /= len(val_loader)

    print(f"[{epoch+1}/{EPOCHS}] loss={total_loss:.4f}, val_acc={val_acc*100:.2f}%")

torch.save(model.state_dict(), "weights/finetuned_efficientnet.pt")
print("✅ Finetuning finished")
