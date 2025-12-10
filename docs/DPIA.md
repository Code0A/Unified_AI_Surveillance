# Data Protection Impact Assessment (DPIA)

This analysis evaluates privacy risks when processing facial images and EEG data.

---

## 1. Data Processed
### Facial Data
- Raw images  
- Cropped faces  
- Embeddings  

### EEG Data
- Raw EEG signals  
- Processed arrays  

---

## 2. Purpose of Data Processing  
- Emotion classification  
- Valence/arousal prediction  
- Training a personalized multimodal model  

---

## 3. Legal Basis  
If used in the EU/UK:
- Explicit consent (GDPR Art. 6 & 9)
- Research exemptions where allowed

---

## 4. Risks  
| Risk | Description |
|------|-------------|
| Identity leakage | Raw faces identifiable |
| Emotional profiling | Sensitive psychological inference |
| Data breach | Exposed biometric or EEG signals |

---

## 5. Mitigations  
### Technical Measures  
- Use **cropped + normalized** images instead of full scenes  
- Store only **embeddings** when possible  
- Enable encryption for EEG files  
- Access control (private GitHub or local storage)

### Organizational Measures  
- Explain data use clearly  
- Restrict access to authorized persons  
- Limit storage time  

---

## 6. Retention & Deletion  
Users can:
- Remove subject embeddings  
- Delete raw images  
- Reset calibration data  

---

## 7. Conclusion  
If implemented with consent, local computation, and minimal storage, the system’s privacy impact is **acceptable** for research and educational use.

