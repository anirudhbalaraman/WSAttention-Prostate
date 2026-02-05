# Contributing

## Running Tests

```bash
# Full test suite
pytest tests/

# Single test
pytest tests/test_run.py::test_run_pirads_training
```

Tests run in `--dry_run` mode (2 epochs, batch_size=2, no W&B logging).

## Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for lint errors
ruff check .

# Auto-format code
ruff format .
```

**Ruff configuration** (from `pyproject.toml`):

| Setting | Value |
|---------|-------|
| Line length | 100 |
| Quote style | Double quotes |
| Rules | E (errors), W (warnings) |
| Ignored | E501 (line too long) |

## SLURM Job Scripts

Job scripts are in `job_scripts/` and are configured for GPU partitions:

```bash
sbatch job_scripts/train_pirads.sh
sbatch job_scripts/train_cspca.sh
```

Key SLURM settings used:

| Setting | Value |
|---------|-------|
| Partition | `gpu` |
| Memory | 128 GB |
| GPUs | 1 |
| Time limit | 48 hours |

!!! tip
    The SLURM job name (`--job-name`) automatically becomes the `run_name`, which determines the log directory at `logs/<run_name>/`.

## Project Conventions

- **Configs** are stored in `config/` as YAML files
- **Logs** are written to `logs/<run_name>/` including TensorBoard events and training logs
- **Models** are saved to `logs/<run_name>/` during training; best models are saved to `models/` for deployment
- **Cache** is stored at `logs/<run_name>/cache/` and cleaned up automatically after training
