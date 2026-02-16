import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

# =========================
# CONFIG
# =========================

INPUT_ROOT = "../dataset/QSLR2024/features/fullFrame-256x256px"
OUTPUT_ROOT = "../dataset/QSLR2024/features/fullFrame-256x256px_faceNetMasked"

BLUR_KERNEL = (51, 51)

# Set None untuk full processing
MAX_FRAMES = None
# MAX_FRAMES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# INIT DETECTOR
# =========================

detector = MTCNN(keep_all=True, device=device)


# =========================
# FACE MASK FUNCTION
# =========================

def mask_face_dynamic(img):
    h, w, _ = img.shape

    boxes, _ = detector.detect(img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            # Clamp ke batas frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            bw = x2 - x1
            bh = y2 - y1

            # Expand kecil agar wajah tertutup natural
            expand_x = int(bw * 0.05)
            expand_y = int(bh * 0.05)

            x1 = max(0, x1 - expand_x)
            x2 = min(w, x2 + expand_x)
            y1 = max(0, y1 - expand_y)
            y2 = min(h, y2 + expand_y)

            bw = x2 - x1
            bh = y2 - y1

            # Buat mask ellipse
            mask = np.zeros_like(img)

            center_x = x1 + bw // 2
            center_y = y1 + bh // 2

            axes_x = int(bw * 0.50)
            axes_y = int(bh * 0.55)

            cv2.ellipse(
                mask,
                (center_x, center_y),
                (axes_x, axes_y),
                0, 0, 360,
                (255, 255, 255),
                -1
            )

            # Blur hanya area ellipse
            blurred_img = cv2.GaussianBlur(img, BLUR_KERNEL, 0)
            img = np.where(mask == 255, blurred_img, img)

    return img


# =========================
# PROCESS SPLIT
# =========================

def process_split(split_name):
    input_split = os.path.join(INPUT_ROOT, split_name)
    output_split = os.path.join(OUTPUT_ROOT, split_name)

    all_images = []

    for root, _, files in os.walk(input_split):
        for file in files:
            if file.lower().endswith(".jpg"):
                all_images.append(os.path.join(root, file))

    if MAX_FRAMES is not None:
        all_images = all_images[:MAX_FRAMES]

    print(f"{split_name}: {len(all_images)} frames")

    for input_path in tqdm(all_images, desc=f"Processing {split_name}"):
        rel_path = os.path.relpath(input_path, input_split)
        output_path = os.path.join(output_split, rel_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img = cv2.imread(input_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_masked = mask_face_dynamic(img_rgb)
        img_masked = cv2.cvtColor(img_masked, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, img_masked)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        process_split(split)

    print("\nDone.")
