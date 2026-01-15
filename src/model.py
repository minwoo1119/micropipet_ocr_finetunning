import torch
import torch.nn as nn
from torchvision import models

def load_efficientnet_b0(num_classes=10, weight_path=None):
    model = models.efficientnet_b0(pretrained=False)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if weight_path:
        ckpt = torch.load(weight_path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)

    return model
