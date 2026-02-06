# Inference

## Full Pipeline

`run_inference.py` runs the complete pipeline: preprocessing followed by PI-RADS classification and csPCa risk prediction.

```bash
python run_inference.py --config config/config_preprocess.yaml
```

This script:

1. Runs all four preprocessing steps (register, segment, histogram match, heatmap)
2. Loads the PI-RADS model from `models/pirads.pt`
3. Loads the csPCa model from `models/cspca_model.pth`
4. For each scan: predicts PI-RADS score, csPCa risk probability, and identifies the top-5 most-attended patches

### Required Model Files

Place these in the `models/` directory:

| File | Description |
|------|-------------|
| `pirads.pt` | Trained PI-RADS MIL model checkpoint |
| `cspca_model.pth` | Trained csPCa model checkpoint |
| `prostate_segmentation_model.pt` | Pre-trained prostate segmentation model |

### Output Format

Results are saved to `<output_dir>/results.json`:

```json
{
    "patient_001.nrrd": {
        "Predicted PIRAD Score": 4.0,
        "csPCa risk": 0.8234,
        "Top left coordinate of top 5 patches(x,y,z)": [
            [32, 45, 7],
            [28, 50, 7],
            [35, 42, 8],
            [30, 48, 6],
            [33, 44, 8]
        ]
    }
}
```

### Label Mapping

PI-RADS predictions are 0-indexed internally and shifted by +2 for display:

| Internal Label | PI-RADS Score |
|---------------|---------------|
| 0 | PI-RADS 2 |
| 1 | PI-RADS 3 |
| 2 | PI-RADS 4 |
| 3 | PI-RADS 5 |

csPCa risk is a continuous probability in [0, 1].

## Testing Individual Models

### PI-RADS Testing

```bash
python run_pirads.py --mode test \
    --config config/config_pirads_test.yaml \
    --checkpoint models/pirads.pt
```

Reports Quadratic Weighted Kappa (QWK) across multiple seeds.

### csPCa Testing

```bash
python run_cspca.py --mode test \
    --config config/config_cspca_test.yaml \
    --checkpoint_cspca models/cspca_model.pth
```

Reports AUC, sensitivity, and specificity with 95% confidence intervals across 20 seeds (default).
