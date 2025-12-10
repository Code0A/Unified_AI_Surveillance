#!/usr/bin/env python3
"""
Realtime inference demo.

Requirements:
  pip install opencv-python facenet-pytorch torch torchvision

Usage example:
  python src/inference/realtime_inference.py \
    --model_ckpt outputs/multimodal/multimodal_best.pt \
    --device cuda \
    --score_threshold 0.5

Notes:
  - If you do not have EEG input, the model will accept zeros for EEG and still predict.
  - Calibration: press "c" while the window is focused to switch into calibration mode.
  - Quit: press "q".
"""
import argparse
import time
import os
from collections import deque

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from pathlib import Path

# relative imports
from src.models.fusion_model import FusionModel

def build_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = FusionModel(num_classes=7).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def prepare_image(mtcnn, frame, transform):
    # mtcnn returns cropped face as PIL if specified, but we'll use bounding boxes here
    try:
        boxes, probs = mtcnn.detect(frame)
    except Exception:
        boxes = None
        probs = None

    if boxes is None or len(boxes) == 0:
        return None, None, None

    # choose largest box
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    box = boxes[idx].astype(int)
    x1,y1,x2,y2 = box
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None, None, None
    try:
        img_t = transform(face).unsqueeze(0)  # [1,3,224,224]
    except Exception:
        return None, None, None
    score = float(probs[idx]) if probs is not None and len(probs)>idx else 1.0
    return img_t, (x1,y1,x2,y2), score

def visualize(frame, box, text):
    if box is not None:
        x1,y1,x2,y2 = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return frame

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)   # face detector
    transform = build_transform()

    print("Loading model:", args.model_ckpt)
    model = load_checkpoint(args.model_ckpt, device)
    print("Model loaded.")

    # If no subject id provided, use 0
    subject_id = torch.tensor([args.subject_id], dtype=torch.long, device=device)

    # Placeholder for EEG: zeros [B, channels, time]. Adjust channels/time to match training preprocess.
    eeg_channels = args.eeg_channels
    eeg_time = args.eeg_time
    eeg_placeholder = torch.zeros((1, eeg_channels, eeg_time), dtype=torch.float32, device=device)

    # Webcam init
    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera index {}".format(args.cam_index))

    # smoothing
    preds_deque = deque(maxlen= args.smooth_len)

    print("Starting realtime inference. Press 'q' to quit. Press 'c' for calibration mode.")
    calibration_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        img_t, box, score = prepare_image(mtcnn, frame, transform)
        display_text = "No face"

        if img_t is not None and score >= args.score_threshold:
            img_t = img_t.to(device)
            # batch dims
            out = model(image=img_t, eeg=eeg_placeholder, subject_id=subject_id)
            logits = out["logits"].detach().cpu().numpy().squeeze()
            probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
            pred_idx = int(np.argmax(probs))
            valence = float(out["valence"].detach().cpu().numpy().squeeze())
            arousal = float(out["arousal"].detach().cpu().numpy().squeeze())
            display_text = f"Pred: {pred_idx} P{probs[pred_idx]:.2f} V{valence:.2f} A{arousal:.2f}"
            preds_deque._

