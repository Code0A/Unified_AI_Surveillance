#!/usr/bin/env python3
"""
preprocess_affectnet.py

AffectNet comes in two folders:
    images/
    labels.csv  (usually contains filepath + expression index 0–7)

This script:
    - Uses MTCNN to crop faces
    - Saves 224x224 processed images
    - Generates new processed CSV

Usage:
    python scripts/preprocess_affectnet.py \
        --img_folder data/raw/affectnet/images \
        --labels_csv data/raw/affectnet/labels.csv \
        --out_folder data/processed/affectnet
"""

import argparse
from facenet_pytorch import MTCNN
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

def ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def crop_face(mtcnn, img):
    try:
        boxes, probs = mtcnn.detect(img)
        if boxes is None: return None
        # largest face
        idx = int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes]))
        x1,y1,x2,y2 = boxes[idx].astype(int)
        face = img[y1:y2, x1:x2]
        if face.size == 0: return None
        return face
    except:
        return None

def main(args):
    device = "cuda" if args.use_cuda else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device)

    df = pd.read_csv(args.labels_csv)
    img_folder = Path(args.img_folder)
    out_folder = Path(args.out_folder)
    ensure(out_folder)
    out_img_folder = out_folder / "images"
    ensure(out_img_folder)

    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="AffectNet preprocess"):
        path = img_folder / row["filepath"]
        img = cv2.imread(str(path))
        if img is None:
            continue

        face = crop_face(mtcnn, img)
        if face is None:
            continue

        face = cv2.resize(face, (224,224))
        save_path = out_img_folder / f"{i}.jpg"
        cv2.imwrite(str(save_path), face)

        rows.append([str(save_path), row["expression"]])

    out_csv = out_folder / "processed_affectnet.csv"
    pd.DataFrame(rows, columns=["filepath", "label"]).to_csv(out_csv, index=False)

    print("AffectNet processing complete.")
    print("Saved:", out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_folder", type=str, required=True)
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--out_folder", type=str, required=True)
    p.add_argument("--use_cuda", action="store_true")
    args = p.parse_args()
    main(args)

