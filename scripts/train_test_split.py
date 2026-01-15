import os
import shutil
import random

SRC_ROOT = "data_balanced"
DST_ROOT = "data"
TRAIN_RATIO = 0.8

for split in ["train", "val"]:
    for label in map(str, range(10)):
        os.makedirs(os.path.join(DST_ROOT, split, label), exist_ok=True)

for label in map(str, range(10)):
    src_dir = os.path.join(SRC_ROOT, label)
    images = os.listdir(src_dir)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    for fname in train_imgs:
        shutil.copy(
            os.path.join(src_dir, fname),
            os.path.join(DST_ROOT, "train", label, fname)
        )

    for fname in val_imgs:
        shutil.copy(
            os.path.join(src_dir, fname),
            os.path.join(DST_ROOT, "val", label, fname)
        )

    print(
        f"[OK] label {label}: "
        f"train={len(train_imgs)}, val={len(val_imgs)}"
    )
