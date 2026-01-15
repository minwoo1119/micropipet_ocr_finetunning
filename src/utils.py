import torch

def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)
