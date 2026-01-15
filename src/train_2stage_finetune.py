import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset
from transforms import train_transform, val_transform
from model import load_efficientnet_b0
from utils import accuracy
from early_stopping import EarlyStopping


# ======================
# Config
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_WORKERS = 4

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 8

LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-5

WEIGHT_PATH = "weights/efficientnet_b0_base.pt"
SAVE_PATH   = "weights/finetuned_efficientnet_b0.pt"

# ======================
# Dataset
# ======================
train_ds = build_dataset("data/train", train_transform)
val_ds   = build_dataset("data/val", val_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ======================
# Model
# ======================
model = load_efficientnet_b0(
    num_classes=10,
    weight_path=WEIGHT_PATH
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

# ==========================================================
# Stage 1: Classifier only
# ==========================================================
print("\n========== Stage 1: Head-only fine-tuning ==========")

for p in model.features.parameters():
    p.requires_grad = False

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_STAGE1
)

for epoch in range(STAGE1_EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"[Stage1][{epoch+1}/{STAGE1_EPOCHS}]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_acc += accuracy(outputs, labels)

    val_acc /= len(val_loader)
    print(f"[Stage1][{epoch+1}] loss={total_loss:.4f}, val_acc={val_acc*100:.2f}%")

# ==========================================================
# Stage 2: Full fine-tuning + Early Stopping
# ==========================================================
print("\n========== Stage 2: Full fine-tuning (Early Stop enabled) ==========")

for p in model.parameters():
    p.requires_grad = True

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR_STAGE2
)

early_stop = EarlyStopping(
    patience=3,       # 3 epoch 연속 개선 없으면 stop
    min_delta=0.001   # 0.1% 이상만 개선으로 인정
)

best_stage2_acc = 0.0

for epoch in range(STAGE2_EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(
        train_loader,
        desc=f"[Stage2][{epoch+1}/{STAGE2_EPOCHS}]"
    ):
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

    print(
        f"[Stage2][{epoch+1}] "
        f"loss={total_loss:.4f}, val_acc={val_acc*100:.2f}%"
    )

    # ------------------
    # Early Stopping
    # ------------------
    if val_acc > best_stage2_acc:
        best_stage2_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print("   🔥 Best Stage2 model saved")

    if early_stop.step(val_acc):
        print(f"\n⏹ Early stopping triggered at epoch {epoch+1}")
        break


# ======================
# Save
# ======================
torch.save(model.state_dict(), SAVE_PATH)
print(f"\n✅ 2-stage fine-tuning finished → {SAVE_PATH}")
