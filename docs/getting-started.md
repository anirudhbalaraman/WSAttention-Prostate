# Getting Started

## Installation

For installation and downloading the saved weights of the models, please check the [github](https://github.com/anirudhbalaraman/WSAttention-Prostate) repository.

## Data Format

Input MRI scans should be in **NRRD** format with three modalities per patient: T2W, DWI, and ADC.

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
    "test": [...]
}
```

Paths are relative to `data_root`. PI-RADS labels are 0-indexed (`0` = PI-RADS 2, `3` = PI-RADS 5). csPCa labels are binary (0 or 1).
