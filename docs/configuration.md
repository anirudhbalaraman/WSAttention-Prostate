# Configuration

## Config System

Configuration follows a three-level hierarchy:

1. **CLI defaults** — Argparse defaults in `run_pirads.py`, `run_cspca.py`, etc.
2. **YAML overrides** — Values from `--config <file>.yaml` override CLI defaults
3. **SLURM job name** — If `SLURM_JOB_NAME` is set, it overrides `run_name`

```bash
# CLI defaults are overridden by YAML config
python run_pirads.py --mode train --config config/config_pirads_train.yaml
```

!!! note
    YAML values **always override** CLI defaults for any key present in the YAML file (`args.__dict__.update(config)`). To override a YAML value, edit the YAML file or omit the key from YAML so the CLI default is used.

## PI-RADS Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | — | `train` or `test` (required) |
| `config` | — | Path to YAML config file |
| `data_root` | — | Root folder of images |
| `dataset_json` | — | Path to dataset JSON file |
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
| `run_name` | `train_pirads` | Run name for logging |
| `dry_run` | `False` | Quick test mode |

## csPCa Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | — | `train` or `test` (required) |
| `config` | — | Path to YAML config file |
| `data_root` | — | Root folder of images |
| `dataset_json` | — | Path to dataset JSON file |
| `num_classes` | `4` | PI-RADS classes (for backbone initialization) |
| `mil_mode` | `att_trans` | MIL algorithm for backbone |
| `tile_count` | `24` | Number of patches per scan |
| `tile_size` | `64` | Patch spatial size |
| `depth` | `3` | Slices per patch |
| `use_heatmap` | `True` | Enable heatmap-guided patch sampling |
| `workers` | `2` | DataLoader workers |
| `checkpoint_pirads` | — | Path to pre-trained PI-RADS model (required for train) |
| `checkpoint_cspca` | — | Path to csPCa checkpoint (required for test) |
| `epochs` | `30` | Max training epochs |
| `batch_size` | `32` | Scans per batch |
| `optim_lr` | `2e-4` | Learning rate |
| `num_seeds` | `20` | Number of random seeds for CI |
| `val_every` | `1` | Validation frequency |
| `dry_run` | `False` | Quick test mode |

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
| `project_dir` | — | Project root (for reference images and models) |

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
    project_dir: /path/to/WSAttention-Prostate
    ```

## Dry-Run Mode

The `--dry_run` flag configures a minimal run for quick testing:

- Epochs: 2
- Batch size: 2
- Workers: 0
- Seeds: 2
- W&B: disabled

```bash
python run_pirads.py --mode train --config config/config_pirads_train.yaml --dry_run
```
