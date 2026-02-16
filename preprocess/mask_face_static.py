import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_ROOT = "../dataset/QSLR2024/features/fullFrame-256x256px"
OUTPUT_ROOT = "../dataset/QSLR2024/features/fullFrame-256x256px_faceMasked"

# Mask region (relative to image size)
# Bisa Anda tweak jika perlu
X1_RATIO = 0.45
X2_RATIO = 0.70
Y1_RATIO = 0.15
Y2_RATIO = 0.38

BLUR_KERNEL = (51, 51)  # harus ganjil


# =========================
# FUNCTION
# =========================

def mask_face_static(img):
    h, w, _ = img.shape

    x1 = int(w * X1_RATIO)
    x2 = int(w * X2_RATIO)
    y1 = int(h * Y1_RATIO)
    y2 = int(h * Y2_RATIO)

    face_region = img[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(face_region, BLUR_KERNEL, 0)
    img[y1:y2, x1:x2] = blurred

    return img


def process_split(split_name):
    input_split = os.path.join(INPUT_ROOT, split_name)
    output_split = os.path.join(OUTPUT_ROOT, split_name)

    for root, dirs, files in os.walk(input_split):
        rel_path = os.path.relpath(root, input_split)
        target_dir = os.path.join(output_split, rel_path)

        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(".jpg"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_dir, file)

                img = cv2.imread(input_path)
                img_masked = mask_face_static(img)
                cv2.imwrite(output_path, img_masked)


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        print(f"Processing {split}...")
        process_split(split)

    print("Done.")
