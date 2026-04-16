# xray-pneumonia-detection-ai
AI-based chest X-ray classification for pneumonia detection with transfer learning, model evaluation, and explainable AI (Grad-CAM).
# 🩺 Chest X-ray Pneumonia Detection using Deep Learning

## 📌 Overview
This project develops a deep learning model to classify chest X-ray images into **NORMAL** and **PNEUMONIA** categories.  
The system integrates model training, evaluation, and explainability to simulate a real-world medical AI workflow.

---

## 🧠 Methodology

### 🔹 Model
- Transfer Learning using **MobileNetV2**
- Custom classification head (Dense layers)
- Fine-tuning of top layers for domain adaptation

### 🔹 Data Processing
- Image resizing and normalization
- Data augmentation (rotation, zoom, flip)
- Handling class imbalance using **class weights**

---

## 📊 Evaluation

Model performance was assessed using:

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

### ⚠️ Key Clinical Insight
- **False Negatives were minimised** (critical in healthcare)
- Some **False Positives remain**, indicating model sensitivity

---

## 🔬 Explainability (Grad-CAM)

Grad-CAM was used to visualise model attention.

### 🔍 Observation:
- Model focuses on **lung regions**
- Attention is **diffuse rather than localised**

### 🧠 Interpretation:
This suggests the model relies on **global intensity patterns** rather than specific pathological features such as focal consolidation.

---


