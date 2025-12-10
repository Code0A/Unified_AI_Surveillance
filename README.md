# Unified AI Surveillance for Personalized Behavioural & Emotional Prediction  
A multimodal AI system combining **deep visual emotion recognition**, **EEG affect prediction**, and **personalized subject modeling** to detect and understand human emotional states in real time.

---

# 📌 Overview  
This repository implements a full research-grade **multimodal affective computing framework** using:

### 🧠 Modalities Used:
- **Facial expressions**  
- **EEG emotional states**  
- **Subject identity embeddings (personalization)**  

### 🔍 The system predicts:
- 7-class emotion classification  
- Continuous valence (1–9)  
- Continuous arousal (1–9)  
- Personalized emotional response model  

This project is designed for **research**, **behavioural science**, **HCI**, **human-aware systems**, and **emotion tracking** — NOT for misuse or surveillance without consent.

---

# 📂 Repository Structure

unified-ai-surveillance/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
│
├── src/
│ ├── datasets/
│ │ ├── fer_loader.py
│ │ ├── deap_loader.py
│ │ ├── affectnet_loader.py
│ │ ├── emotion_kaggle_loader.py
│ │ └── unified_dataset.py
│ │
│ ├── models/
│ │ ├── visual_backbone.py
│ │ ├── eeg_backbone.py
│ │ ├── fusion_model.py
│ │ └── personalization.py
│ │
│ ├── train/
│ │ ├── train_visual_only.py
│ │ └── train_multimodal.py
│ │
│ ├── inference/
│ │ ├── realtime_inference.py
│ │ └── calibration.py
│ │
│ └── utils/
│ ├── metrics.py
│ └── samplers.py
│
├── scripts/
│ ├── preprocess_faces.py
│ ├── preprocess_deap.py
│ └── preprocess_affectnet.py
│
└── docs/
├── MODEL_CARD.md
├── ETHICS.md
├── DPIA.md
└── DATASET_GUIDE.md# 🔧 Features

### 🎥 **Visual Emotion Recognition**
- Uses a ResNet18 backbone  
- Trained on FER2013, AffectNet, Kaggle Emotions  
- Predicts 7 emotions

### 🧠 **EEG-based Affective Computing**
- Uses DEAP dataset  
- Predicts valence & arousal  
- 1D CNN for EEG processing  

### 🔄 **Multimodal Fusion Model**
- Combines:
  - image embedding (256-dim)
  - EEG embedding (128-dim)
  - subject embedding (64-dim)
- Predicts:
  - emotion class  
  - valence  
  - arousal  

### 👤 **Personalization Layer**
- Trainable subject embedding for each user  
- Calibration routine lets a new user adapt the model to their own emotion patterns

### 🖥 **Realtime Emotion Prediction**
- Webcam face detection using MTCNN  
- Live smoothing & prediction confidence  
- On-the-fly user calibration  

---

# 🛠 Installation

Make sure Python 3.10+ is installed.

Install dependencies:

```bash
pip install -r requirements.txt📦 Dataset Preprocessing

You MUST preprocess datasets before training.

FER2013:
python scripts/preprocess_faces.py \
    --src_folder data/raw/fer2013_images \
    --labels_csv data/raw/fer2013.csv \
    --out_folder data/processed/fer2013

AffectNet:
python scripts/preprocess_affectnet.py \
    --img_folder data/raw/affectnet/images \
    --labels_csv data/raw/affectnet/labels.csv \
    --out_folder data/processed/affectnet

DEAP:
python scripts/preprocess_deap.py \
    --raw_folder data/raw/deap/ \
    --out_folder data/processed/deap/

🧪 Training
👉 1. Train Visual Model Only
python src/train/train_visual_only.py \
    --fer_csv data/processed/fer2013.csv \
    --epochs 10 \
    --batch_size 64 \
    --out_dir outputs/visual

👉 2. Train Full Multimodal Fusion Model
python src/train/train_multimodal.py \
    --fer_csv data/processed/fer2013.csv \
    --deap_eeg data/processed/deap/eeg \
    --deap_labels data/processed/deap/labels.csv \
    --epochs 20 \
    --batch_size 32 \
    --out_dir outputs/multimodal

🖥 Realtime Inference

Run live detection from webcam:

python src/inference/realtime_inference.py \
    --model_ckpt outputs/multimodal/multimodal_best.pt \
    --device cuda

Controls:
Key	Action
q	Quit
c	Start calibration
k	Capture sample during calibration
👤 User Calibration Mode

Calibration improves accuracy for new users.

Steps:

Press c during realtime inference

Look directly into the camera

Press k to capture each labeled sample

Enter the emotion label (0–6) in the terminal

The system creates:

a new subject embedding

a calibrated model checkpoint

📘 Documentation

All docs are in /docs:

MODEL_CARD.md — complete model explanation

ETHICS.md — misuse prevention, risk analysis

DPIA.md — privacy protection compliance

DATASET_GUIDE.md — dataset descriptions + preprocessing

⚠️ Limitations

Not medically approved

Can misclassify strong lighting/occlusions

EEG predictions vary depending on hardware

Dataset biases influence accuracy

Should not be used for enforcement, surveillance, HR, or legal systems

🤝 Contributing

Contributions are welcome!Submit a pull request or open an issue.

📜 License

MIT License (recommended for research + open use)

✨ Author

Built by Aradhyea Saroha
With complete architecture design, training pipeline, documentation & inference system.


---

If you want:
✅ A cleaner minimal README  
or  
✅ A research-paper formatted README  
or  
✅ A GitHub Pages site version  

Just tell me **“Make README minimal / research style / GitHub Pages style”** and I’ll generate it.


