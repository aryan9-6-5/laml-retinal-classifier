---
title: LAML Retinal Classifier
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 🔬 LAML — Lesion-Aware Multi-Task Learning for Retinal Disease Classification

> Multi-label fundus image classifier trained on 6 datasets using ConvNeXtTiny + Squeeze-and-Excitation attention, dual-head architecture, and focal loss with lesion-aware auxiliary supervision.

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| Backbone | ConvNeXtTiny (ImageNet pretrained) |
| Attention | Squeeze-and-Excitation (SE) channel block |
| Heads | Disease head (8 classes) + Lesion concept head |
| Loss | Focal BCE + λ·Lesion BCE (λ=0.3) |
| Input Size | 512 × 512 px |
| Output | Multi-label sigmoid (8 diseases) |

---

## 📊 Performance

| Metric | Score |
|---|---|
| Macro AUC | **0.8490** |
| Rare Disease AUC | **0.8752** |
| Thresholds | Per-class F1-optimal |

---

## 🗂️ Training Datasets

| Dataset | Disease Focus |
|---|---|
| ODIR-5K | 8-class multi-label (primary) |
| EyePACS | Diabetic Retinopathy |
| ORIGA | Glaucoma |
| AREDS2 | Age-related Macular Degeneration |
| RFMiD | Multi-label retinal diseases |
| Hypertension Fundus | Hypertensive Retinopathy |

---

## 🏥 Disease Classes

| Code | Disease |
|---|---|
| N | Normal |
| D | Diabetic Retinopathy |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration |
| H | Hypertensive Retinopathy |
| M | Myopia |
| O | Other Findings |

---

## ⚙️ Training Pipeline

- **Phase 1** — 15 epochs, backbone frozen, heads trained at LR=1e-3
- **Phase 2** — Fine-tune top 40% of backbone at LR≈9e-7
- **Augmentation** — Flip, rotate, brightness/contrast/saturation/hue, random crop
- **Imbalance handling** — Class-weighted focal loss + sample weights
- **Split** — Patient-level GroupShuffleSplit (no data leakage between paired eyes)
- **Threshold tuning** — Per-class F1-optimal on validation set

---

## 🚀 How to Use

1. Upload a fundus photograph (JPG or PNG)
2. The model preprocesses it with CLAHE enhancement and resizes to 512×512
3. Predictions are shown with probability bars and Grad-CAM attention maps
4. Switch to the **Performance** tab to explore ROC curves, confusion matrices, and AUC scores
5. Switch to the **Dataset** tab to explore the training data distribution

---

## 📁 Repository Structure
```
├── app.py                  # Streamlit application
├── Dockerfile              # Docker deployment config
├── requirements.txt        # Python dependencies
└── LAML_outputs/
    ├── LAML_final.keras    # Trained model (109MB, Git LFS)
    ├── thresholds.json     # Per-class F1-optimal thresholds
    ├── config.json         # Model configuration
    ├── auc_scores_balanced.csv
    ├── ablation_results.csv
    ├── training_history.png
    ├── ROC_curves.png
    └── dataset_manifest.csv
```

---

## 📄 Citation

If you use this work, please cite:
```
@misc{laml2024,
  title     = {LAML: Lesion-Aware Multi-Task Learning for Retinal Disease Classification},
  year      = {2024},
  note      = {ConvNeXtTiny backbone, multi-dataset training, ODIR-5K label space}
}
```

---

*Built with TensorFlow · ConvNeXtTiny · Streamlit · Deployed on Hugging Face Spaces*
