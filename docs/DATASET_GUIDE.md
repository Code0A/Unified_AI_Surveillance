# Dataset Guide

This project integrates **multiple emotion recognition datasets** into a unified framework.

---

# 1. FER2013  
- 48×48 grayscale images  
- 7 emotion classes  
- CSV format  

### Preprocessing  
Run:
```bash
python scripts/preprocess_faces.py --src_folder data/raw/fer2013_images --labels_csv data/raw/fer2013.csv --out_folder data/processed/fer2013
3. Emotion Kaggle Dataset

Folder-based emotions

Diverse faces

Handled by:
emotion_kaggle_loader.py

4. DEAP

32-channel EEG recordings

Labels: valence, arousal, dominance, liking

Preprocess DEAP
python scripts/preprocess_deap.py \
  --raw_folder data/raw/deap/ \
  --out_folder data/processed/deap/


Outputs:

eeg/*.npy

labels.csv

5. CASE Dataset

Multimodal affective reactions

Optional extra dataset

Use similar preprocessing as AffectNet.

6. Unified Dataset

All datasets mapped into:

{
  "image": tensor  (or zeros),
  "eeg": tensor    (or zeros),
  "label_cls": int,
  "valence": float,
  "arousal": float,
  "subject_id": int
}


Used by:

src/datasets/unified_dataset.py

7. Notes

Missing modalities are handled by zeros

Always run preprocessing before training

Use balanced samplers to reduce bias


---

# 🎉 **F6 COMPLETE**

You now have **full professional documentation** for GitHub — ready to publish the project publicly.

---

# 👉 FINAL STEP: F7 (Root Files)
If you want, I will now provide the final remaining files:

✔ `requirements.txt`  
✔ `.gitignore`  
✔ `Dockerfile`  

Just say **“SURE — F7”** and I will generate them.
