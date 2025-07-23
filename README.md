# BrainTumour-Analysis-Pipeline
A complete deep learning pipeline for brain tumor analysis, integrating ViT-based classification, YOLOv8 object detection, risk level estimation, and survival prediction. The system automates tumor detection from MRI scans, classifies tumor presence, estimates severity, and supports clinical insights.
# Brain Tumor Analysis Pipeline 🧠

This project implements a **modular pipeline** for brain tumor analysis using deep learning techniques, including classification, object detection, risk assessment, and survival prediction.

## 🔍 Overview

The pipeline consists of the following stages:

1. **Tumor Classification (ViT)** – Determines whether an image shows presence of a tumor using a Vision Transformer (ViT).
2. **Tumor Detection (YOLOv8)** – Locates tumor regions in the image with bounding boxes.
3. **Risk Classification** – Calculates tumor area and categorizes risk as `Low`, `Medium`, or `High`.
4. **Survival Estimation (Optional)** – Placeholder for future integration based on clinical features.

---

## 📁 Project Structure
├── classification/
│ └── vit_train.py, vit_infer.py
├── detection/
│ └── yolo_detect.py
├── risk_analysis/
│ └── risk_label_gen.py, train_risk_model.py
├── models/
│ └── vit_brain_tumor.pth (Not included)
├── data/ (Not included)
│ └── train/, valid/, test/
├── runs/ (Generated outputs - ignored by Git)
├── requirements.txt
├── README.md
└── .gitignore

yaml

---

## 🚀 Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/AneeshS24/brain-tumor-pipeline.git
   cd brain-tumor-pipeline

