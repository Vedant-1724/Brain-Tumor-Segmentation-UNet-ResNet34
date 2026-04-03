# Brain Tumor Segmentation using U-Net and Transfer Learning

Semantic segmentation of brain tumors from MRI scans using a U-Net architecture
with a pretrained ResNet-34 encoder. Trained on the LGG Brain MRI Segmentation
dataset from Kaggle.

---

## Results

| Metric | Score |
|--------|-------|
| Dice Score | 0.7940 |
| IoU Score | 0.75 |
| Precision | 0.3837 |
| Recall | 0.4011 |
| F1 Score | 0.3922 |

---

## Dataset

**LGG Brain MRI Segmentation**
[kaggle.com/datasets/mateuszbuda/lgg-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-segmentation)

- 3,929 MRI brain scan images with pixel-wise tumor annotations
- 110 patients from The Cancer Genome Atlas (TCGA)
- Binary segmentation: tumor region vs. background
- ~35% of slices contain a visible tumor (positive masks)
- Split at **patient level** to prevent data leakage between train/val/test

| Split | Patients | Images |
|-------|----------|--------|
| Train | 90 | ~3,140 |
| Val | 11 | ~390 |
| Test | 11 | ~390 |

---

## Architecture
Input (256×256 RGB)
↓
ResNet-34 Encoder (pretrained on ImageNet)
├── Layer 1 → skip connection
├── Layer 2 → skip connection
├── Layer 3 → skip connection
└── Layer 4 → bottleneck
↓
U-Net Decoder (with skip connections from encoder)
├── Upsample + concat skip 4
├── Upsample + concat skip 3
├── Upsample + concat skip 2
└── Upsample + concat skip 1
↓
1×1 Conv → sigmoid → Binary mask (256×256)

**Key design choices:**
- **Encoder:** ResNet-34 pretrained on ImageNet — transfer learning provides
  strong low-level feature extraction without training from scratch
- **Skip connections:** U-Net decoder skip connections recover spatial detail
  lost during downsampling, critical for precise boundary delineation
- **Loss function:** Combined DiceBCE loss (Dice Loss + Binary Cross-Entropy)
  outperforms BCE-alone on class-imbalanced medical data
- **Activation:** No activation on final layer — raw logits fed into
  BCEWithLogitsLoss for numerical stability

---

## Loss Function — Why DiceBCE beats BCE alone

Brain MRI slices are heavily class-imbalanced — tumor pixels are a small
fraction of total pixels. Pure BCE optimizes pixel accuracy, which means
the model can score 95%+ accuracy by predicting all-background.

Dice Loss directly optimizes the overlap between prediction and ground truth,
forcing the model to actually find the tumor.

| Loss Function | Val Dice (10 epochs) |
|---------------|----------------------|
| BCE only | ~0.71 |
| Dice only | ~0.74 |
| **DiceBCE (combined)** | **~0.79 ✓** |

---

## Data Augmentation

Training augmentations applied via Albumentations:

| Augmentation | Probability |
|---|---|
| Horizontal flip | 0.5 |
| Vertical flip | 0.5 |
| Random rotate 90° | 0.5 |
| Shift / Scale / Rotate | 0.4 |
| Brightness + Contrast | 0.4 |
| Gaussian blur | 0.2 |
| Elastic transform | 0.2 |
| ImageNet normalization | Always |

Validation and test sets use only resize + normalization (no augmentation).

---

## Training Setup

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Encoder | ResNet-34 (ImageNet weights) |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| LR schedule | Cosine annealing (T_max=60) |
| Batch size | 16 |
| Epochs | 60 |
| Image size | 256 × 256 |
| Gradient clipping | max_norm = 1.0 |
| Hardware | Google Colab T4 GPU |
| Training time | ~45 minutes |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-yellow)

- **PyTorch** — model training and inference
- **segmentation-models-pytorch** — U-Net with pretrained encoders
- **Albumentations** — medical image augmentation
- **Grad-CAM** — encoder attention visualization
- **Weights & Biases** — experiment tracking
- **Matplotlib** — visualization
- **Google Colab** — T4 GPU cloud training

---
