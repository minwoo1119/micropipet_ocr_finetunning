import os
import shutil
import random

SRC_ROOT = "data_raw"      # 원본 (0~9)
DST_ROOT = "data_balanced" # 결과
TARGET_PER_CLASS = 800

os.makedirs(DST_ROOT, exist_ok=True)

for label in map(str, range(10)):
    src_dir = os.path.join(SRC_ROOT, label)
    dst_dir = os.path.join(DST_ROOT, label)
    os.makedirs(dst_dir, exist_ok=True)

    images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(images) < TARGET_PER_CLASS:
        raise RuntimeError(
            f"[ERROR] label {label}: {len(images)} images < {TARGET_PER_CLASS}"
        )

    selected = random.sample(images, TARGET_PER_CLASS)

    for fname in selected:
        shutil.copy(
            os.path.join(src_dir, fname),
            os.path.join(dst_dir, fname)
        )

    print(f"[OK] label {label}: {TARGET_PER_CLASS} images")
