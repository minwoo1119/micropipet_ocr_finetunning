from torchvision.datasets import ImageFolder

def build_dataset(root, transform):
    return ImageFolder(root=root, transform=transform)
