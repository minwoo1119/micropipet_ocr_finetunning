import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from transforms import val_transform
from model import load_efficientnet_b0
from utils import accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = build_dataset("data/val", val_transform)
loader = DataLoader(dataset, batch_size=32)

model = load_efficientnet_b0(num_classes=10)
model.load_state_dict(torch.load("weights/finetuned_efficientnet.pt"))
model.to(DEVICE)
model.eval()

acc = 0
with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        acc += accuracy(outputs, labels)

acc /= len(loader)
print(f"Validation Accuracy: {acc*100:.2f}%")
