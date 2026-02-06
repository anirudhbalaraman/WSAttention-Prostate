# Preprocessing Reference

## Overview

Preprocessing is orchestrated by `preprocess_main.py`, which runs steps in sequence. Each step receives and returns the `args` namespace, updating directory paths as it goes.

## Step Dependencies

| Step | Requires |
|------|----------|
| `register_and_crop` | — |
| `get_segmentation_mask` | `register_and_crop` |
| `histogram_match` | `register_and_crop`, `get_segmentation_mask` |
| `get_heatmap` | `register_and_crop`, `get_segmentation_mask`, `histogram_match` |

Dependencies are validated at runtime — the pipeline will exit with an error if steps are out of order.

## register_files

```python
def register_files(args) -> args
```

Registers and crops T2, DWI, and ADC images to a standardized spacing and size.

**Process:**

1. Reads images from `args.t2_dir`, `args.dwi_dir`, `args.adc_dir`
2. Resamples to spacing `(0.4, 0.4, 3.0)` mm using `picai_prep.Sample`
3. Center-crops with `args.margin` (default 0.2) in x/y dimensions
4. Saves to `<output_dir>/t2_registered/`, `DWI_registered/`, `ADC_registered/`

**Updates `args`:** `t2_dir`, `dwi_dir`, `adc_dir` → registered directories.

## get_segmask

```python
def get_segmask(args) -> args
```

Generates prostate segmentation masks from T2W images using a pre-trained model.

**Process:**

1. Loads model config from `<project_dir>/config/inference.json`
2. Loads checkpoint from `<project_dir>/models/prostate_segmentation_model.pt`
3. Applies MONAI transforms: orientation (RAS), spacing (0.5 mm isotropic), intensity normalization
4. Runs inference and inverts transforms to original space
5. Post-processes: retains only top 10 slices by non-zero voxel count
6. Saves NRRD masks to `<output_dir>/prostate_mask/`

**Updates `args`:** adds `seg_dir`.

## histmatch

```python
def histmatch(args) -> args
```

Matches the intensity histogram of each modality to a reference image.

**Process:**

1. Reads reference images from `<project_dir>/dataset/` (`t2_reference.nrrd`, `dwi_reference.nrrd`, `adc_reference.nrrd`, `prostate_segmentation_reference.nrrd`)
2. For each patient, matches histograms within the prostate mask using `skimage.exposure.match_histograms`
3. Saves to `<output_dir>/t2_histmatched/`, `DWI_histmatched/`, `ADC_histmatched/`

**Updates `args`:** `t2_dir`, `dwi_dir`, `adc_dir` → histogram-matched directories.

### get_histmatched

```python
def get_histmatched(
    data: np.ndarray,
    ref_data: np.ndarray,
    mask: np.ndarray,
    ref_mask: np.ndarray,
) -> np.ndarray
```

Low-level function that performs histogram matching on masked regions only. Unmasked pixels remain unchanged.

## get_heatmap

```python
def get_heatmap(args) -> args
```

Generates combined DWI/ADC attention heatmaps.

**Process:**

1. For each file, reads DWI, ADC, and prostate mask
2. Computes DWI heatmap: `(dwi - min) / (max - min)` within mask
3. Computes ADC heatmap: `(max - adc) / (max - min)` within mask (inverted — low ADC = high attention)
4. Combines via element-wise multiplication
5. Re-normalizes to [0, 1]
6. Saves to `<output_dir>/heatmaps/`

**Updates `args`:** adds `heatmapdir`.

!!! info "Edge cases"
    If all values within the mask are identical for a modality (DWI or ADC), that modality's heatmap is skipped. If both are constant, the heatmap defaults to all ones.
