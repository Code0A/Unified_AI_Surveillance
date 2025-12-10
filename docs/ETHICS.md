# Ethics & Responsible Use

This project deals with **emotion recognition**, which is a sensitive topic. Emotional inference systems can influence personal autonomy, mental health, and privacy.

---

# 1. Key Ethical Principles

### 🔸 1.1 Transparency  
Users must be informed:
- what the model does  
- what data it uses  
- how predictions are made  

### 🔸 1.2 Consent  
When collecting facial or EEG data:
- obtain explicit consent  
- allow users to revoke data  

### 🔸 1.3 Privacy  
- Do not store raw images unnecessarily  
- Use anonymized embeddings instead of identity labels  
- Encryption recommended for EEG data  

### 🔸 1.4 No Harm  
The model **must not** be used to:
- Diagnose psychological disorders  
- Make employment or legal decisions  
- Monitor individuals without consent  

---

# 2. Dataset Bias  
Datasets such as FER2013 contain:
- Imbalanced emotion distributions  
- Limited cultural diversity  
- Low resolution images  

These biases may result in:
- Lower accuracy for certain demographics  
- Misinterpretation of affect  

Balanced sampling & calibration help reduce this.

---

# 3. Misuse Scenarios  
❌ Predict emotions for workplace punishment  
❌ Mass-surveillance without consent  
❌ Medical diagnosis  

---

# 4. Recommended Safe-Use  
✔ Research  
✔ Human-computer interaction  
✔ Affective computing studies  
✔ Emotion-aware interfaces  

---

# 5. Calibration Ethics  
Calibration should be:
- Voluntary  
- Transparent  
- Private (no cloud uploads)  

---

# 6. Contact  
For ethical/abuse concerns, contact the project maintainer.

