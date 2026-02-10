# Configuration

YAML values always override CLI defaults (`args.__dict__.update(config)`). To override a YAML value, edit the YAML file or omit the key so the CLI default is used.

```bash
python run_pirads.py --mode train --config config/config_pirads_train.yaml
```

## PI-RADS Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | — | `train` or `test` (required) |
| `config` | — | Path to YAML config file |
| `data_root` | — | Root folder of T2W images |
| `dataset_json` | — | Path to dataset JSON file. Format should be as specified in [Getting Started](getting-started.md) |
| `num_classes` | `4` | Number of output classes (PI-RADS 2–5) |
| `mil_mode` | `att_trans` | MIL algorithm (`mean`, `max`, `att`, `att_trans`, `att_trans_pyramid`) |
| `tile_count` | `24` | Number of patches per scan |
| `tile_size` | `64` | Patch spatial size in pixels |
| `depth` | `3` | Number of slices per patch |
| `use_heatmap` | `True` | Enable heatmap-guided patch sampling |
| `workers` | `2` | DataLoader workers |
| `checkpoint` | `None` | Path to resume from checkpoint |
| `epochs` | `50` | Max training epochs |
| `early_stop` | `40` | Epochs without improvement before stopping |
| `batch_size` | `4` | Scans per batch |
| `optim_lr` | `3e-5` | Base learning rate |
| `weight_decay` | `0` | Optimizer weight decay |
| `amp` | `False` | Enable automatic mixed precision |
| `val_every` | `1` | Validation frequency (epochs) |
| `wandb` | `False` | Enable Weights & Biases logging |
| `project_name` | `Classification_prostate` | W&B project name |
| `run_name` | `train_pirads` | Run name for logging. If using SLURM, takes SLURM JOB_ID |

## csPCa Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | — | `train` or `test` (required) |
| `config` | — | Path to YAML config file |
| `data_root` | — | Root folder of images |
| `dataset_json` | — | Path to dataset JSON file |
| `checkpoint_pirads` | — | Path to pre-trained PI-RADS model (required for train) |
| `checkpoint_cspca` | — | Path to csPCa checkpoint (required for test) |
| `epochs` | `30` | Max training epochs |
| `batch_size` | `32` | Scans per batch |
| `optim_lr` | `2e-4` | Learning rate |
| `num_seeds` | `20` | Number of random seeds for CI |

Shared parameters (`num_classes`, `mil_mode`, `tile_count`, `tile_size`, `depth`, `use_heatmap`, `workers`, `val_every`) have the same defaults as PI-RADS.

## Preprocessing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | — | Path to YAML config file |
| `steps` | — | Steps to execute (required, one or more) |
| `t2_dir` | — | Directory of T2W images |
| `dwi_dir` | — | Directory of DWI images |
| `adc_dir` | — | Directory of ADC images |
| `seg_dir` | — | Directory of segmentation masks |
| `output_dir` | — | Output directory |
| `margin` | `0.2` | Center-crop margin fraction |

## Example YAML

=== "PI-RADS Training"

    ```yaml
    data_root: /path/to/registered/t2_hist_matched
    dataset_json: /path/to/PI-RADS_data.json
    num_classes: 4
    mil_mode: att_trans
    tile_count: 24
    tile_size: 64
    depth: 3
    use_heatmap: true
    workers: 4
    epochs: 100
    batch_size: 8
    optim_lr: 2e-4
    weight_decay: 1e-5
    amp: true
    wandb: true
    ```

=== "csPCa Training"

    ```yaml
    data_root: /path/to/registered/t2_hist_matched
    dataset_json: /path/to/csPCa_data.json
    num_classes: 4
    mil_mode: att_trans
    tile_count: 24
    tile_size: 64
    depth: 3
    use_heatmap: true
    workers: 6
    checkpoint_pirads: /path/to/models/pirads.pt
    epochs: 80
    batch_size: 8
    optim_lr: 2e-4
    ```

=== "Preprocessing"

    ```yaml
    t2_dir: /path/to/raw/t2
    dwi_dir: /path/to/raw/dwi
    adc_dir: /path/to/raw/adc
    output_dir: /path/to/processed
    ```
