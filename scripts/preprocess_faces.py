#!/usr/bin/env python3
"""
preprocess_faces.py

Use MTCNN to crop faces from raw images for FER2013, AffectNet, or Kaggle emotion datasets.
Outputs clean 224x224 face crops and a CSV with paths + labels.

Example:
    python scripts/preprocess_faces.py \
        --src_folder data/raw/fer2013_images \
        --labels_csv data/raw/fer2013.csv \
        --out_folder data/processed/fer2013 \
        --has_labels True
"""

import os
import argparse
import pandas as pd
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def crop_face(mtcnn, img):
    try:
        boxes, probs = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return None
        # Get largest face
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        box = boxes[idx].astype(int)
        x1, y1, x2, y2 = box
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return face
    except Exception:
        return None

def main(args):
    device = "cuda" if args.use_cuda else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device)

    src_folder = Path(args.src_folder)
    out_folder = Path(args.out_folder)
    ensure_dir(out_folder)

    if args.has_labels:
        df = pd.read_csv(args.labels_csv)
    else:
        df = pd.DataFrame({"filepath": sorted([str(p) for p in src_folder.glob("*.*")]),
                           "label": [-1] * len(list(src_folder.glob("*.*")))})

    processed_rows = []
    out_img_folder = out_folder / "images"
    ensure_dir(out_img_folder)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Cropping faces"):
        img_path = src_folder / row["filepath"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        face = crop_face(mtcnn, img)
        if face is None:
            continue

        face = cv2.resize(face, (224, 224))
        new_path = out_img_folder / f"{i}.jpg"
        cv2.imwrite(str(new_path), face)
        processed_rows.append([str(new_path), row["label"]])

    out_csv = out_folder / "processed_faces.csv"
    pd.DataFrame(processed_rows, columns=["filepath", "label"]).to_csv(out_csv, index=False)
    print(f"Preprocessing done. Output saved to: {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_folder", type=str, required=True)
    p.add_argument("--labels_csv", type=str, default="")
    p.add_argument("--out_folder", type=str, required=True)
    p.add_argument("--has_labels", type=bool, default=True)
    p.add_argument("--use_cuda", action="store_true")
    args = p.parse_args()
    main(args)

