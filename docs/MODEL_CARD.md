
```markdown
# Model Card — Unified AI Surveillance Model

## 1. Model Summary  
The Unified AI Surveillance model is a **multimodal neural architecture** that predicts:

- **Discrete emotions** (7-class FER)
- **Valence** (1–9 continuous)
- **Arousal** (1–9 continuous)

### Architecture Components  
| Component | Description |
|----------|-------------|
| Visual Backbone | ResNet18 → 256-dim embedding |
| EEG Backbone | 1D CNN → 128-dim embedding |
| Personalization | Subject embedding 64-dim |
| Fusion Layer | Concatenation → FC2048 → classifier/regressors |
| Outputs | logits, valence, arousal |

---

## 2. Intended Use  
- Behavioural research  
- Human–computer interaction  
- Assistive emotional monitoring  
- Wellbeing & mental health analytics  
- Personalized affect prediction  

Not intended as a **clinical** or **security-only** tool.

---

## 3. Training Data  
| Dataset | Modality | Purpose |
|---------|----------|---------|
| FER2013 | Faces | Emotion classification |
| AffectNet | Faces | High-quality emotion labels |
| Kaggle Emotion | Faces | Extra image diversity |
| DEAP | EEG | Valence/Arousal regression |
| CASE | Additional affective cues | Optional |

---

## 4. Metrics  
- **Accuracy** (classification)  
- **Macro-F1** (balanced emotion scoring)  
- **RMSE/MAE** (VA regression)  

---

## 5. Ethical Considerations  
- Potential privacy risks  
- Bias from dataset imbalance  
- Misinterpretation of emotional output  
- Not suitable for medical or legal decisions  

See full **ETHICS.md**.

---

## 6. Limitations  
- Faces under extreme angles may fail  
- EEG inputs require specific hardware  
- Valence/arousal accuracy varies by subject  
- Calibration needed for best performance  

---

## 7. Authors  
Aradhyea Saroha + contributors

