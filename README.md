# R_SANTE
Classification of mitotic phases in histopathology data

Mines Paris - PSL - CBIO

# Mitosis Classification under Uncertainty: A Soft-Label Approach

**Authors:** Elyesse & Alexandre  
**Supervision:** Thomas & Raphaël  
**Context:** Digital Pathology / MIDOG Challenge

---

## Abstract

Mitotic count is a critical prognostic factor in breast cancer grading. However, classifying mitotic phases (*Prophase, Metaphase, Anaphase, Telophase*) is subject to high inter-observer variability.

**The Problem:** Standard Deep Learning approaches use "Hard Labels" (one image = one class), forcing a definitive truth even when experts disagree. This discards valuable biological ambiguity.

**Our Solution:** We propose a **Soft-Label Training Strategy**. Instead of forcing a binary decision, our model learns a probability distribution derived from multiple expert annotations. We use a **ConvNeXt-Tiny** architecture to capture subtle morphological features and evaluate performance using a novel stratified protocol (Consensus vs. Disagreement).

---

## Key Features

* **Soft-Labeling Pipeline:** Dynamically generates probabilistic targets based on expert agreement ratios.
* **State-of-the-Art Backbone:** Uses `ConvNeXt-Tiny` pre-trained on ImageNet, finetuned for histopathology.
* **Stratified Evaluation:**
    * **Group A (Consensus):** Standard Accuracy.
    * **Group B (Disagreement):** "Double Coverage" Accuracy (Top-2 predictions must match both experts).
* **Advanced Visualization:**
    * **Probabilistic Confusion Matrices** (Mass distribution analysis).
    * **Confidence Calibration** (KDE Plots).
    * **Latent Space t-SNE** colored by model uncertainty.

---

## Project Structure

```bash
├── data/                  # Dataset and CSV metadata
├── src/
│   ├── config.py          # Global hyperparameters (Batch size, LR, etc.)
│   ├── dataset.py         # Custom PyTorch Dataset with Soft-Label logic
│   ├── train.py           # Training loop with Best Model saving
│   ├── evaluation.py      # Stratified metrics (Group A/B, Top-2)
│   ├── visualization.py   # t-SNE, Confusion Matrices, GradCAM
│   └── utils.py           # Helper functions
├── r_sante_v5_soft_label.ipynb  # MAIN NOTEBOOK (Run this)
└── README.md

```

---

## Installation & Usage

1. **Clone the repository:**
```bash
git clone
cd r_sante

```


2. **Install dependencies:**
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy opencv-python torchcam

```


3. **Run the Main Notebook:**
Open `r_sante_v5_soft_label.ipynb` in Jupyter Lab or VS Code. This notebook contains the full pipeline:
* Data Loading & EDA
* Training (ConvNeXt)
* Full Evaluation Battery
* Visualizations



---

## Results & Performance

Our model successfully captures the "Human Baseline" and models biological uncertainty.

| Metric | Result | Interpretation |
| --- | --- | --- |
| **Human Agreement** | **~80.5%** | Theoretical upper bound (Expert 1 vs Expert 2). |
| **Model Accuracy (Top-1)** | **~87.0%** | The model matches human expert performance. |
| **Indulgent Accuracy (Top-2)** | **~95.5%** | The correct phase is almost always in the top 2 choices. |
| **Calibration** | **Success** | KDE plots confirm high confidence on consensus, low confidence on disagreement. |

### Visual Interpretation

The latent space projection (t-SNE) reveals that **uncertainty is structural**:

* **Green Circles (Confident):** Located at cluster centers (Prototypical phases).
* **Red Crosses (Uncertain):** Located at the boundaries between clusters (Transitional phases).

---

## Perspectives

* **Active Learning:** Use the model's uncertainty to query experts only on ambiguous samples.
* **Temporal Consistency:** Integrate video data to enforce the biological sequence (Prophase  Telophase).
* **Clinical Correlation:** Investigate if the "uncertainty score" itself is a prognostic biomarker for tumor aggressiveness.

---

## License

This project is intended for academic and research purposes.
