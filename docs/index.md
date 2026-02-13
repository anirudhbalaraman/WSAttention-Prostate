<div style="text-align: center; margin-bottom: 2em;">
  <img src="assets/logo.svg" alt="WSAttention-Prostate Logo" width="560">
</div>

# WSAttention-Prostate

**Weakly Supervised Attention-Based Deep Learning for Prostate Cancer Characterization from Bi-Parametric Prostate MRI.**

WSAttention-Prostate is a two-stage deep learning pipeline that predicts clinically significant prostate cancer (csPCa) risk and PI-RADS score (2 to 5) from T2W, DWI, and ADC bpMRI sequences. The backbone is a patch based 3D Multiple-Instance Learning (MIL) model pre-trained to classify PI-RADS scores and fine-tuned to predict csPCa risk — all without requiring lesion-level annotations.

💡 **GUI for real-time inference available at [Hugging Face Spaces](https://huggingface.co/spaces/anirudh0410/WSA_Prostate)**

## Key Features

- **Weakly-supervised attention** — Heatmap-guided patch sampling and cosine-similarity attention loss replace the need for voxel-level labels
- **3D Multiple Instance Learning** — Extracts volumetric patches from MRI scans and aggregates them via transformer + attention pooling
- **Two-stage pipeline** — Stage 1 trains a 4-class PI-RADS classifier; Stage 2 freezes its backbone and trains a binary csPCa head
- **Preprocessing** — Preprocessing to minimize inter-center MRI acquisiton variability.
- **End-to-end pipeline** — Registration, segmentation, histogram matching, and heatmap generation, and inferencing in a single configurable pipeline

## Pipeline Overview

```mermaid
%%{init: {'themeVariables': { 'fontSize': '20px' }}}%%
flowchart LR
    A[Raw bpMRI</br>T2 + DWI + ADC] --> B[Preprocessing]
    B --> C[Stage 1:</br>PI-RADS Classification]
    C --> D[Stage 2:</br>csPCa Prediction]
    D --> E[Risk Score + Top-5 Salient Patches]
```

## Quick Links

- [Getting Started](getting-started.md) — Installation and first run
- [Pipeline](pipeline.md) — Full walkthrough of preprocessing, training, and evaluation
- [Architecture](architecture.md) — Model design and tensor shapes
- [Configuration](configuration.md) — YAML config reference
