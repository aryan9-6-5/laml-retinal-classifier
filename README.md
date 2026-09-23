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

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20CPU-FF6F00.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20Research-lightgrey.svg)](#disclaimer)

> **LAML** is a multi-label retinal disease classification system trained across **6 ophthalmic datasets** (>80,000 fundus photographs). Built with a **ConvNeXtTiny** backbone enhanced by **Squeeze-and-Excitation (SE)** channel attention, LAML employs a dual-head multi-task architecture with **auxiliary lesion supervision** to achieve state-of-the-art diagnostic performance and visual explainability via Grad-CAM.

---

## 📌 Table of Contents
- [Highlights & Benchmark Performance](#-highlights--benchmark-performance)
- [Model Architecture](#-model-architecture)
- [Disease Classes & Decision Thresholds](#-disease-classes--decision-thresholds)
- [Auxiliary Lesion Concepts](#-auxiliary-lesion-concepts)
- [Image Preprocessing & Pipeline](#-image-preprocessing--pipeline)
- [Visual Explainability (Grad-CAM)](#-visual-explainability-grad-cam)
- [Datasets & Patient-Level Splitting](#-datasets--patient-level-splitting)
- [Streamlit Web Interface](#-streamlit-web-interface)
- [Installation & Quickstart](#-installation--quickstart)
- [Docker Deployment](#-docker-deployment)
- [Programmatic Python Inference](#-programmatic-python-inference)
- [Repository & Artifacts Structure](#-repository--artifacts-structure)
- [Clinical & Regulatory Disclaimer](#-clinical--regulatory-disclaimer)
- [Citation](#-citation)

---

## 🏆 Highlights & Benchmark Performance

- **Macro AUC**: **`0.8490`** across all 8 ODIR disease categories on balanced test splits.
- **Rare Disease AUC**: **`0.8752`**, demonstrating high sensitivity on underrepresented ophthalmic pathologies.
- **Auxiliary Supervision**: Trained with a 25-concept auxiliary lesion head ($\lambda = 0.3$), constraining intermediate representations to attend to clinically valid lesions (e.g., drusen, flame hemorrhages, cotton wool spots).
- **Patient-Level Leakage Prevention**: Split using `GroupShuffleSplit` on patient IDs so paired left/right eyes never cross train and evaluation boundaries.
- **F1-Optimal Calibrated Thresholds**: Class-specific decision boundaries tuned on validation data to maximize F1-score rather than assuming a naive 0.5 cutoff.

| Metric | Score | Validation Protocol |
|---|---|---|
| **Macro AUC** | **0.8490** | Balanced test set evaluation across 8 classes |
| **Rare Disease AUC** | **0.8752** | Evaluated on rare conditions (A, H, M, O) |
| **Input Resolution** | 512 × 512 px | Lanczos4 interpolation + CLAHE contrast equalization |
| **Primary Label Space** | 8 Multi-label classes | ODIR-5K taxonomy with sigmoid outputs |
| **Auxiliary Label Space**| 25 Clinical lesions | Multi-task BCE loss ($\lambda = 0.3$) |

---

## 🧠 Model Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │          Input Fundus Photograph             │
                  │   512 × 512 × 3 RGB (CLAHE Y-channel prepped)│
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            ConvNeXtTiny Backbone             │
                  │            (ImageNet Pre-trained)            │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │        Squeeze-and-Excitation (SE)           │
                  │          Channel Attention Block             │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │           Shared Dense Embeddings            │
                  └──────────────┬────────────────┬──────────────┘
                                 │                │
            ┌────────────────────┴───┐        ┌───┴─────────────────────┐
            │   Primary Disease Head │        │  Auxiliary Lesion Head  │
            │     8-Class Sigmoid    │        │    25-Concept Sigmoid   │
            │      (Loss: Focal BCE) │        │     (Loss: λ · BCE)     │
            └────────────────────────┘        └─────────────────────────┘
```

### Component Details
1. **Backbone**: `ConvNeXtTiny` pre-trained on ImageNet. A modern pure-convolutional network offering Transformer-like receptive field and performance with convolutional compute efficiency.
2. **Channel Attention**: A **Squeeze-and-Excitation (SE)** block recalibrates channel-wise feature responses adaptively, magnifying features corresponding to pathology markers.
3. **Dual Heads**:
   - **Disease Head**: 8 independent sigmoid outputs corresponding to ODIR-5K diagnostic classes.
   - **Lesion Concept Head**: 25 independent sigmoid outputs predicting specific lesion tokens extracted from clinical notes.
4. **Loss Function**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}}(y_{\text{disease}}, \hat{y}_{\text{disease}}; \gamma=2, \alpha_c) + \lambda \cdot \mathcal{L}_{\text{BCE}}(y_{\text{lesion}}, \hat{y}_{\text{lesion}})$$
   where $\lambda = 0.3$, and $\alpha_c$ applies class-weighted sample penalties based on inverse frequency to counter class imbalance.
5. **Two-Phase Training Protocol**:
   - **Phase 1** (15 epochs): Backbone weights frozen; shared dense and dual prediction heads trained at $\text{LR} = 10^{-3}$.
   - **Phase 2** (Fine-tuning): Top 40% of the ConvNeXtTiny backbone unfrozen with cosine decay learning rate down to $\text{LR} \approx 9 \times 10^{-7}$.

---

## 🏥 Disease Classes & Decision Thresholds

Because retinal diseases exhibit severe natural class imbalance, evaluating predictions against a default `0.5` threshold leads to poor sensitivity for rare conditions. LAML uses **F1-optimal decision thresholds** tuned on the validation set:

| Code | Disease Class | Description & Key Clinical Signs | Optimal Threshold ($\tau$) |
|:---:|---|---|:---:|
| **N** | **Normal** | Healthy fundus, clear optical media, sharp disc margins, healthy macula | **0.43** |
| **D** | **Diabetic Retinopathy** | Microaneurysms, blot/flame hemorrhages, hard/soft exudates, neovascularization | **0.55** |
| **G** | **Glaucoma** | Optic disc cupping (increased cup-to-disc ratio), neuroretinal rim thinning | **0.55** |
| **C** | **Cataract** | Media opacification, diffuse blurring, yellowish haze obscuring retinal detail | **0.85** |
| **A** | **Age-related MD** | Macular drusen, geographic atrophy, choroidal neovascularization (CNV) | **0.49** |
| **H** | **Hypertension** | Arteriolar narrowing, arteriovenous (AV) nicking, copper/silver wiring | **0.47** |
| **M** | **Myopia** | Pathological myopia, peripapillary chorioretinal atrophy, tessellation | **0.43** |
| **O** | **Other Findings** | Retinal vein occlusions (BRVO/CRVO), epiretinal membranes, macular holes | **0.63** |

*A disease condition is positively diagnosed if $\hat{y}_c \ge \tau_c$. Multiple positive flags can be triggered simultaneously.*

---

## 🔍 Auxiliary Lesion Concepts

The auxiliary lesion head supervises 25 clinically documented fundus patterns:

```
• diabetic retinopathy         • macular degeneration        • glaucoma
• cataract                     • hypertensive retinopathy    • myopia
• drusen                       • hemorrhage                  • exudate
• cotton wool spot             • neovascularization          • disc cupping
• retinal detachment           • laser photocoagulation      • macular hole
• epiretinal membrane          • vitreous opacity            • branch retinal vein occlusion (BRVO)
• central retinal vein (CRVO)  • retinitis pigmentosa        • morning glory syndrome
• coloboma                     • tessellation                • pathological myopia
• macular edema
```

This multi-task supervision anchors feature activations to authentic physiological findings, dramatically improving model interpretability and generalizability.

---

## 🧪 Image Preprocessing & Pipeline

Input images are processed through a standardized biomedical vision pipeline:

1. **Decoding**: Image bytes decoded to BGR format.
2. **Color Space Conversion**: BGR converted to **YUV** color space to isolate luminance ($Y$) from chromaticity ($U, V$).
3. **CLAHE Enhancement**:
   - Contrast Limited Adaptive Histogram Equalization is applied to the **$Y$ channel only**:
     ```python
     clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(15, 15))
     yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
     ```
   - Prevents chromatic distortion while accentuating subtle micro-lesions, hemorrhages, and fine vessel boundaries.
4. **Resizing**: Converted back to BGR and downsampled/upsampled to **$512 \times 512$** using high-order **Lanczos4 interpolation** (`cv2.INTER_LANCZOS4`).
5. **Normalization**: Transformed to RGB and scaled to `[0.0, 1.0]` as `float32`.

---

## 🗺️ Visual Explainability (Grad-CAM)

LAML integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visually substantiate its predictions:

- **Target Layer**: Activations are extracted from the final 4D convolutional feature map in the ConvNeXtTiny backbone.
- **Backpropagation**: Gradients of the predicted class score with respect to the feature map are computed using `tf.GradientTape`.
- **Channel Weighting**: Gradients are globally pooled across spatial dimensions to calculate importance weights for each feature channel.
- **Heatmap Generation**: Positive linear combinations (ReLU) of weighted feature maps are normalized to $[0, 1]$.
- **Overlay**: The heatmap is upscaled to $512 \times 512$, colorized with the `jet` colormap, and blended with the original fundus photograph at opacity $\alpha = 0.42$.

---

## 🗂️ Datasets & Patient-Level Splitting

LAML aggregates over **80,000 fundus images** across six benchmark clinical repositories:

| Dataset | Sample Count | Primary Pathology & Focus |
|---|---|---|
| **ODIR-5K** | ~10,000 | 8-class multi-label standard benchmark (Primary) |
| **EyePACS** | ~50,000 | Diabetic Retinopathy screening grades (0–4) |
| **ORIGA** | ~650 | Glaucoma diagnosis with cup-to-disc annotations |
| **AREDS2** | ~10,000 | Age-Related Macular Degeneration (AMD) stages |
| **RFMiD** | ~3,200 | Diverse multi-label retinal findings & external test split |
| **Hypertension Fundus** | ~1,200 | Hypertensive retinopathy grading |

### Leakage Prevention
Retinal studies typically feature two images per patient (OD: right eye, OS: left eye). Random splitting inadvertently leaks systemic patient traits into test partitions. LAML strictly utilizes **`GroupShuffleSplit` grouped by patient identifier** to ensure that both eyes of any subject reside exclusively within the training, validation, or test partition.

---

## 💻 Streamlit Web Interface

The repository provides a dashboard (`app.py`) structured into 4 tabs:

1. **Inference**:
   - Drag-and-drop fundus photograph upload (JPG/PNG).
   - Real-time probability prediction bars with F1-optimal threshold indicators.
   - Visual diagnosis pills (e.g. `✓ Normal`, `Diabetic Retinopathy`, `Glaucoma`).
   - Dynamic Grad-CAM heatmaps for the highest-probability pathological classes.
   - Expandable raw confidence, threshold, and margin table.
2. **Performance**:
   - Per-class AUC metrics on balanced test sets.
   - Ablation study table comparing backbone variations and auxiliary loss weighting.
   - Phase 1 & Phase 2 loss/AUC training curves.
   - Multi-class ROC curves and confusion matrices.
   - Active decision threshold registry.
3. **Dataset**:
   - Total sample count and source distribution charts.
   - Prevalence breakdown across the 8 diagnostic labels.
   - Patient-level split validation statistics.
   - Manifest browser to explore metadata and patient records.
4. **About**:
   - Architectural specifications and training summary.
   - Automated file presence audit confirming available weights and assets.

---

## ⚡ Installation & Quickstart

### Prerequisites
- Python **3.10** or higher
- Git and Git LFS (`git lfs install`)

### 1. Clone the Repository
```bash
git clone https://huggingface.co/spaces/aryanhanumakonda/laml-retinal-classifier
cd laml-retinal-classifier
git lfs pull
```

### 2. Set Up a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Linux environments**: OpenCV requires X11/GLib libraries. Install them via your package manager if not already present:
> ```bash
> sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
> ```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

The application includes a production-ready `Dockerfile` based on `python:3.10-slim`:

### Build Docker Image
```bash
docker build -t laml-retinal-classifier:latest .
```

### Run Container
```bash
docker run -d -p 7860:7860 --name laml-app laml-retinal-classifier:latest
```
Access the application at `http://localhost:7860`.

---

## 🐍 Programmatic Python Inference

To integrate LAML into your own Python pipelines without the Streamlit UI:

```python
import cv2
import json
import numpy as np
import tensorflow as tf

# 1. Load Model & Thresholds
model = tf.keras.models.load_model("LAML_outputs/LAML_final.keras", compile=False)
with open("LAML_outputs/thresholds.json", "r") as f:
    thresholds = json.load(f)

DISEASE_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
DISEASE_FULL = {
    'N': 'Normal', 'D': 'Diabetic Retinopathy', 'G': 'Glaucoma',
    'C': 'Cataract', 'A': 'Age-related MD', 'H': 'Hypertension',
    'M': 'Myopia', 'O': 'Other Findings'
}

# 2. Preprocessing Function
def preprocess_image(img_path, size=512):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Luminance CLAHE enhancement
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(15, 15))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    
    # Resize and scale
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# 3. Perform Inference
input_tensor = preprocess_image("path_to_fundus_photo.jpg")
predictions = model.predict(input_tensor, verbose=0)

# Extract disease head probabilities
probs = predictions['disease'][0] if isinstance(predictions, dict) else predictions[0]

# 4. Apply Optimal Thresholds
results = {}
for i, code in enumerate(DISEASE_NAMES):
    prob = float(probs[i])
    thr = float(thresholds.get(code, 0.5))
    results[DISEASE_FULL[code]] = {
        "probability": round(prob, 4),
        "threshold": round(thr, 4),
        "positive": prob >= thr
    }

print(json.dumps(results, indent=2))
```

---

## 📁 Repository & Artifacts Structure

```
laml-retinal-classifier/
├── .streamlit/
│   └── config.toml             # Streamlit server and upload configuration
├── Dockerfile                  # Container definition (Python 3.10-slim, port 7860)
├── README.md                   # Project documentation and HF Space card
├── requirements.txt            # Python dependencies (Streamlit, TF, OpenCV, etc.)
├── app.py                      # Interactive Streamlit application
└── LAML_outputs/
    ├── LAML_final.keras        # Trained dual-head model (Git LFS)
    ├── config.json             # Model configuration & lesion vocabulary
    ├── thresholds.json         # F1-optimal per-class decision thresholds
    ├── last_epoch_p2.txt       # Training phase tracking checkpoint
    ├── auc_scores_balanced.csv # Per-class test set AUC scores (Optional/Generated)
    ├── ablation_results.csv    # Architecture ablation results (Optional/Generated)
    ├── dataset_manifest.csv    # Combined multi-dataset manifest (Optional/Generated)
    ├── training_history.png    # Phase 1 & Phase 2 loss/AUC plot (Optional/Generated)
    ├── roc_curves_balanced.png # Receiver Operating Characteristic curves (Optional/Generated)
    └── confusion_matrices_balanced.png # Confusion matrix grid (Optional/Generated)
```

---

## ⚠️ Clinical & Regulatory Disclaimer

> [!CAUTION]
> **FOR INVESTIGATIONAL AND RESEARCH USE ONLY.**
>
> LAML is an experimental machine learning system developed for academic research and technology demonstration.
> - **Not a Medical Device**: LAML has not been cleared, approved, or certified by the U.S. Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other medical regulatory authority.
> - **Not for Clinical Diagnosis**: LAML output must **never** be used as a substitute for professional ophthalmic examination, clinical diagnosis, or medical treatment planning.
> - **Diagnostic Consultation**: All retinal fundus interpretations must be performed by licensed ophthalmologists or qualified healthcare providers using certified clinical equipment.

---

## 📄 Citation

If you use the LAML architecture, code, or methodology in your academic research, please cite:

```bibtex
@misc{laml2024,
  author    = {Aryan Hanumakonda},
  title     = {LAML: Lesion-Aware Multi-Task Learning for Retinal Disease Classification},
  year      = {2024},
  publisher = {Hugging Face},
  journal   = {Hugging Face Spaces},
  howpublished = {\url{https://huggingface.co/spaces/aryanhanumakonda/laml-retinal-classifier}},
  note      = {ConvNeXtTiny backbone with Squeeze-and-Excitation attention and auxiliary lesion supervision}
}
```

---

*Developed with TensorFlow · ConvNeXtTiny · Streamlit · Deployed on Hugging Face Spaces*
