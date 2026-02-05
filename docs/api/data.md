# Data Loading Reference

## get_dataloader

```python
def get_dataloader(args, split: Literal["train", "test"]) -> DataLoader
```

Creates a PyTorch DataLoader with MONAI transforms and persistent caching.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `args` | Namespace with `dataset_json`, `data_root`, `tile_size`, `tile_count`, `depth`, `use_heatmap`, `batch_size`, `workers`, `dry_run`, `logdir` |
| `split` | `"train"` or `"test"` |

**Behavior:**

- Loads data lists from a MONAI decathlon-format JSON
- In `dry_run` mode, limits to 8 samples
- Uses `PersistentDataset` with cache stored at `<logdir>/cache/<split>/`
- Training split is shuffled; test split is not
- Uses `list_data_collate` to stack patches into `[B, N, C, D, H, W]`

## Transform Pipeline

Two variants depending on `args.use_heatmap`:

### With Heatmaps (default)

| Step | Transform | Description |
|------|-----------|-------------|
| 1 | `LoadImaged` | Load T2, mask, DWI, ADC, heatmap (ITKReader, channel-first) |
| 2 | `ClipMaskIntensityPercentilesd` | Clip T2 intensity to [0, 99.5] percentiles within mask |
| 3 | `ConcatItemsd` | Stack T2 + DWI + ADC → 3-channel image |
| 4 | `NormalizeIntensity_customd` | Z-score normalize per channel using mask-only statistics |
| 5 | `ElementwiseProductd` | Multiply mask * heatmap → `final_heatmap` |
| 6 | `RandWeightedCropd` | Extract N patches weighted by `final_heatmap` |
| 7 | `EnsureTyped` | Cast labels to float32 |
| 8 | `Transposed` | Reorder image dims for 3D convolution |
| 9 | `DeleteItemsd` | Remove intermediate keys (mask, dwi, adc, heatmap) |
| 10 | `ToTensord` | Convert to PyTorch tensors |

### Without Heatmaps

| Step | Transform | Description |
|------|-----------|-------------|
| 1 | `LoadImaged` | Load T2, mask, DWI, ADC |
| 2 | `ClipMaskIntensityPercentilesd` | Clip T2 intensity to [0, 99.5] percentiles within mask |
| 3 | `ConcatItemsd` | Stack T2 + DWI + ADC → 3-channel image |
| 4 | `NormalizeIntensityd` | Standard channel-wise normalization (MONAI built-in) |
| 5 | `RandCropByPosNegLabeld` | Extract N patches from positive (mask) regions |
| 6 | `EnsureTyped` | Cast labels to float32 |
| 7 | `Transposed` | Reorder image dims |
| 8 | `DeleteItemsd` | Remove intermediate keys |
| 9 | `ToTensord` | Convert to tensors |

## list_data_collate

```python
def list_data_collate(batch: Sequence) -> dict
```

Custom collation function that stacks per-patient patch lists into batch tensors.

Each sample from the dataset is a list of N patch dictionaries. This function:

1. Stacks `image` across patches: `[N, C, D, H, W]` per sample
2. Stacks `final_heatmap` if present
3. Applies PyTorch's `default_collate` to form the batch dimension

Result: `{"image": [B, N, C, D, H, W], "label": [B], ...}`

## Custom Transforms

### ClipMaskIntensityPercentilesd

```python
ClipMaskIntensityPercentilesd(
    keys: KeysCollection,
    mask_key: str,
    lower: float | None,
    upper: float | None,
    sharpness_factor: float | None = None,
    channel_wise: bool = False,
    dtype: DtypeLike = np.float32,
)
```

Clips image intensity to percentiles computed only from the **masked region**. Supports both hard clipping (default) and soft clipping (via `sharpness_factor`).

### NormalizeIntensity_customd

```python
NormalizeIntensity_customd(
    keys: KeysCollection,
    mask_key: str,
    subtrahend: NdarrayOrTensor | None = None,
    divisor: NdarrayOrTensor | None = None,
    nonzero: bool = False,
    channel_wise: bool = False,
    dtype: DtypeLike = np.float32,
)
```

Z-score normalization where mean and standard deviation are computed only from **masked voxels**. Supports channel-wise normalization.

### ElementwiseProductd

```python
ElementwiseProductd(
    keys: KeysCollection,
    output_key: str,
)
```

Computes the element-wise product of two arrays from the data dictionary and stores the result in `output_key`. Used to combine the prostate mask with the attention heatmap.

## Dataset JSON Format

The pipeline expects a MONAI decathlon-format JSON file:

```json
{
    "train": [
        {
            "image": "relative/path/to/t2.nrrd",
            "dwi": "relative/path/to/dwi.nrrd",
            "adc": "relative/path/to/adc.nrrd",
            "mask": "relative/path/to/mask.nrrd",
            "heatmap": "relative/path/to/heatmap.nrrd",
            "label": 2
        }
    ],
    "test": [...]
}
```

Paths are relative to `data_root`. The `heatmap` key is only required when `use_heatmap=True`.
