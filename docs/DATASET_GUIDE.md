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

