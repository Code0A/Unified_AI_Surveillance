#!/usr/bin/env python3
"""
Simple calibration routine to quickly fine-tune subject embedding.

Flow:
  - Collect N labeled frames for the new subject (user types label for each sample).
  - Fine-tune only the SubjectEmbedding weights for a new subject id (or update an existing id).
  - Return the assigned subject_id.

Notes:
  - The fusion model must have an attribute `subject_embed` which is an nn.Embedding.
  - This routine will append a new embedding at the end of the embedding matrix when allocating
    a new subject_id (naive approach).
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from pathlib import Path

def collect_samples(model, device, transform, mtcnn, cap, eeg_placeholder, n_samples=20):
    """
    Display frames and ask the user to input the label for each captured sample.
    Returns list of (image_tensor, label_int)
    """
    collected = []
    print(f"Collecting {n_samples} labeled frames. For each frame, type emotion label index (0-6) then Enter.")
    idx = 0
    while idx < n_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        img_t = None
        try:
            boxes, probs = mtcnn.detect(frame)
        except Exception:
            boxes = None
            probs = None
        if boxes is None or len(boxes) == 0:
            cv2.imshow("Calibration - No face", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        # choose largest box
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        box_ind = int(np.argmax(areas))
        x1,y1,x2,y2 = boxes[box_ind].astype(int)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        img_t = transform(face).unsqueeze(0).to(device)
        display = frame.copy()
        cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(display, f"Sample {idx+1}/{n_samples} - press 'k' to capture", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("k"):
            # show captured face and ask for label in terminal
            cv2.imshow("Captured", face)
            print("Enter label (0=angry,1=disgust,2=fear,3=happy,4=sad,5=surprise,6=neutral): ")
            label_str = input().strip()
            try:
                label = int(label_str)
                if label < 0 or label > 6:
                    print("Invalid label. Try again.")
                    continue
            except Exception:
                print("Invalid input. Try again.")
                continue
            collected.append((img_t.cpu(), label))
            idx += 1
        elif key == ord("q"):
            print("Calibration interrupted by user.")
            break
    cv2.destroyWindow("Calibration")
    return collected

def allocate_subject_id(model, device):
    """
    Allocate a new subject id by expanding the embedding matrix by 1 and initializing new vector.
    Returns new subject_id (int).
    """
    emb = model.subject_embed.embed  # nn.Embedding
    old_num, dim = emb.weight.shape
    new_num = old_num + 1
    new_weight = torch.cat([emb.weight.data, emb.weight.data.new_zeros((1, dim))], dim=0)
    emb.weight = nn.Parameter(new_weight)
    # initialize new vector small random
    with torch.no_grad():
        emb.weight[-1].normal_(mean=0.0, std=0.01)
    return old_num  # index of newly allocated

def fine_tune_embedding(model, device, samples, eeg_placeholder, subject_id, steps=50, lr=1e-2):
    """
    Fine-tune only the embedding vector for the new subject_id.
    samples: list of (img_tensor_on_cpu, label_int)
    """
    model.train()
    emb = model.subject_embed.embed
    emb.to(device)
    # Create optimizer for embedding only
    opt = torch.optim.SGD([emb.weight], lr=lr)  # we'll mask updates to only subject_id row
    criterion = nn.CrossEntropyLoss()

    # create small batch tensors
    imgs = torch.cat([s[0] for s in samples], dim=0).to(device)   # [N,3,224,224]
    labels = torch.tensor([s[1] for s in samples], dtype=torch.long, device=device)

    # training loop: only update the row corresponding to subject_id
    for step in range(steps):
        opt.zero_grad()
        # forward
        out = model(image=imgs, eeg=eeg_placeholder.repeat(imgs.size(0),1,1).to(device), subject_id=torch.tensor([subject_id]*imgs.size(0), device=device))
        logits = out["logits"]
        loss = criterion(logits, labels)
        loss.backward()

        # zero grads for other embedding rows (mask)
        with torch.no_grad():
            if emb.weight.grad is not None:
                mask = torch.ones_like(emb.weight.grad, dtype=torch.bool)
                mask[subject_id] = False
                emb.weight.grad[mask] = 0.0

        opt.step()
        if step % 10 == 0:
            print(f"[Calib] step {step}/{steps} loss={loss.item():.4f}")

    print("Calibration fine-tuning finished.")
    model.eval()


def run_calibration(model, device, transform, mtcnn, cap, eeg_placeholder, n_samples=20):
    """
    Top-level entry. Returns new_subject_id (int) or None on failure.
    """
    print("Starting calibration routine...")
    samples = collect_samples(model, device, transform, mtcnn, cap, eeg_placeholder, n_samples=n_samples)
    if len(samples) == 0:
        print("No samples collected.")
        return None

    # allocate new subject id
    new_id = allocate_subject_id(model, device)
    print(f"Allocated new subject_id = {new_id}")

    # fine-tune embedding
    fine_tune_embedding(model, device, samples, eeg_placeholder, subject_id=new_id, steps=50, lr=1e-2)

    # Save model locally (optional)
    try:
        ckpt_path = Path("outputs") / "calibrated_model.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict()}, ckpt_path)
        print(f"Calibrated model saved to {ckpt_path}")
    except Exception as e:
        print("Failed to save calibrated model:", e)

    return new_id

