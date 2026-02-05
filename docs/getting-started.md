# Getting Started

## Prerequisites

- Python 3.11+
- NVIDIA GPU recommended (CUDA-compatible)
- ~128 GB RAM for training (configurable via batch size)

## Installation

```bash
git clone https://github.com/ai-assisted-healthcare/WSAttention-Prostate.git
cd WSAttention-Prostate
pip install -r requirements.txt
```

### External Git Dependencies

Two packages are installed directly from GitHub repositories:

| Package | Source | Purpose |
|---------|--------|---------|
| `AIAH_utility` | `ai-assisted-healthcare/AIAH_utility` | Healthcare imaging utilities |
| `grad-cam` | `jacobgil/pytorch-grad-cam` | Gradient-weighted class activation maps |

These are included in `requirements.txt` and install automatically.

## Verify Installation

Run the test suite in dry-run mode:

```bash
pytest tests/
```

Tests use `--dry_run` mode internally (2 epochs, batch_size=2, no W&B).

## Data Format

Input MRI scans should be in **NRRD** or **NIfTI** format with three modalities per patient:

- T2-weighted (T2W)
- Diffusion-weighted imaging (DWI)
- Apparent diffusion coefficient (ADC)

### Dataset JSON Structure

The data pipeline uses MONAI's decathlon-format JSON:

```json
{
    "train": [
        {
            "image": "path/to/t2.nrrd",
            "dwi": "path/to/dwi.nrrd",
            "adc": "path/to/adc.nrrd",
            "mask": "path/to/prostate_mask.nrrd",
            "heatmap": "path/to/heatmap.nrrd",
            "label": 0
        }
    ],
    "test": [
        ...
    ]
}
```

The `image` key points to the T2W image, which serves as the reference modality. Labels for PI-RADS are 0-indexed: label `0` = PI-RADS 2, label `3` = PI-RADS 5. For csPCa, labels are binary (0 or 1).

## Project Structure

```
WSAttention-Prostate/
├── run_pirads.py              # PI-RADS training/testing entry point
├── run_cspca.py               # csPCa training/testing entry point
├── run_inference.py           # Full inference pipeline
├── preprocess_main.py         # Preprocessing entry point
├── config/                    # YAML configuration files
│   ├── config_pirads_train.yaml
│   ├── config_pirads_test.yaml
│   ├── config_cspca_train.yaml
│   ├── config_cspca_test.yaml
│   └── config_preprocess.yaml
├── src/
│   ├── model/
│   │   ├── MIL.py             # MILModel_3D — core MIL architecture
│   │   └── csPCa_model.py     # csPCa_Model + SimpleNN head
│   ├── data/
│   │   ├── data_loader.py     # MONAI data pipeline
│   │   └── custom_transforms.py
│   ├── train/
│   │   ├── train_pirads.py    # PI-RADS training loop
│   │   └── train_cspca.py     # csPCa training loop
│   ├── preprocessing/
│   │   ├── register_and_crop.py
│   │   ├── prostate_mask.py
│   │   ├── histogram_match.py
│   │   └── generate_heatmap.py
│   └── utils.py
├── job_scripts/               # SLURM job templates
├── tests/
├── dataset/                   # Reference images for histogram matching
└── models/                    # Pre-trained model checkpoints
```
