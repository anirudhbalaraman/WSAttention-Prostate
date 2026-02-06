import argparse

import pytest
import torch

from src.data.custom_transforms import NormalizeIntensity_custom
from src.data.data_loader import get_dataloader
from src.model.cspca_model import CSPCAModel
from src.model.mil import MILModel3D
from src.train import train_cspca, train_pirads
from src.train.train_pirads import get_attention_scores


@pytest.fixture
def mock_args():
    # Mocking argparse for the device
    args = argparse.Namespace()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def test_get_attention_scores_logic(mock_args):
    # Setup: 2 samples, 4 patches, images of size 8x8
    batch_size = 2
    num_patches = 4

    # Sample 0: Target = 3 (Cancer), Sample 1: Target = 0 (PI-RADS 2)
    data = torch.randn(batch_size, num_patches, 1, 8, 8)
    target = torch.tensor([3.0, 0.0])

    # Create heatmaps: Sample 0 has one "hot" patch
    heatmap = torch.zeros(batch_size, num_patches, 1, 8, 8)
    heatmap[0, 0] = 10.0  # High attention on patch 0 for the first sample
    heatmap[1, :] = 5.0  # Should be overridden by PI-RADS 2 logic anyway

    att_labels, shuffled_images = get_attention_scores(data, target, heatmap, mock_args)

    # --- TEST 1: Normalization ---
    sums = att_labels.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones(batch_size).to(mock_args.device))

    # --- TEST 2: PI-RADS 2 Uniformity ---
    pirads_2_scores = att_labels[1]
    expected_uniform = torch.ones(num_patches).to(mock_args.device) / num_patches
    torch.testing.assert_close(pirads_2_scores, expected_uniform)

    # --- TEST 4: Output Shapes ---
    assert att_labels.shape == (batch_size, num_patches)
    assert shuffled_images.shape == data.shape


def test_shuffling_consistency(mock_args):
    # Verify that the image and label are shuffled with the SAME permutation
    num_patches = 10

    # Distinct data per patch: [0, 1, 2, 3...]
    data = torch.arange(num_patches).view(1, num_patches, 1, 1, 1).float()
    target = torch.tensor([3.0])

    # Heatmap matches the data indices so we can track the "label"
    heatmap = torch.arange(num_patches).view(1, num_patches, 1, 1, 1).float()

    att_labels, shuffled_images = get_attention_scores(data, target, heatmap, mock_args)

    idx = (shuffled_images[0, :, 0, 0, 0] == 9.0).nonzero(as_tuple=True)[0]
    # The attention score at that same index should be the maximum
    assert att_labels[0, idx] == att_labels[0].max()

    idx = (shuffled_images[0, :, 0, 0, 0] == 0.0).nonzero(as_tuple=True)[0]
    # The attention score at that same index should be the minimum
    assert att_labels[0, idx] == att_labels[0].min()

    shuffled_images = shuffled_images.cpu().squeeze()  # Shape [10]
    att_labels = att_labels.cpu().squeeze()  # Shape [10]

    sorted_vals, original_indices = torch.sort(shuffled_images)
    sorted_labels = att_labels[original_indices]

    for i in range(len(sorted_labels) - 1):
        assert sorted_labels[i] <= sorted_labels[i + 1], (
            f"Alignment broken at index {i}: Image val {sorted_vals[i]} has higher label than {sorted_vals[i + 1]}"
        )


def test_normalize_intensity_custom_masked_stats():
    """
    Test that statistics (mean/std) are calculated ONLY from the masked region,
    but applied to the whole image.
    """

    img = torch.zeros((2, 4, 4), dtype=torch.float32)
    mask = torch.zeros((1, 4, 4), dtype=torch.float32)

    img[0, :, :] = 100.0
    img[0, 0, 0] = 10.0
    img[0, 0, 1] = 20.0

    img[1, :, :] = 50.0
    img[1, 0, 0] = 2.0
    img[1, 0, 1] = 4.0

    mask[0, 0, 0] = 1
    mask[0, 0, 1] = 1

    normalizer = NormalizeIntensity_custom(nonzero=False, channel_wise=True)
    out = normalizer(img, mask)

    assert torch.isclose(out[0, 0, 0], torch.tensor(-1.0)), "Ch0 masked value 1 incorrect"
    assert torch.isclose(out[0, 0, 1], torch.tensor(1.0)), "Ch0 masked value 2 incorrect"
    assert torch.isclose(out[0, 1, 1], torch.tensor(17.0)), "Ch0 background normalization incorrect"

    assert torch.isclose(out[1, 0, 0], torch.tensor(-1.0)), "Ch1 masked value 1 incorrect"
    assert torch.isclose(out[1, 1, 1], torch.tensor(47.0)), "Ch1 background normalization incorrect"


def test_normalize_intensity_constant_area():
    """
    Test edge case where the area under the mask has 0 variance (constant value).
    Std should default to 1.0 to avoid division by zero.
    """
    img = torch.ones((1, 4, 4)) * 10.0  # All values are 10
    mask = torch.ones((1, 4, 4))

    normalizer = NormalizeIntensity_custom(channel_wise=True)
    out = normalizer(img, mask)
    assert torch.allclose(out, torch.zeros_like(out))

    data = torch.rand(1, 10, 10)
    mask = torch.randint(0, 2, (1, 10, 10)).float()
    normalizer = NormalizeIntensity_custom(nonzero=False, channel_wise=True)
    out = normalizer(data, mask)

    masked = data[mask != 0]
    mean_val = torch.mean(masked.float())
    std_val = torch.std(masked.float(), unbiased=False)

    epsilon = 1e-8
    normalized_data = (data - mean_val) / (std_val + epsilon)

    torch.testing.assert_close(out, normalized_data)


def test_run_models():
    args = argparse.Namespace()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.epochs = 1
    args.batch_size = 2
    args.tile_size = 10
    args.tile_count = 5
    args.use_heatmap = True
    args.amp = False
    args.num_classes = 4
    args.dry_run = True
    args.depth = 3

    model = MILModel3D(num_classes=args.num_classes, mil_mode="att_trans")
    model.to(args.device)
    params = model.parameters()
    loader = get_dataloader(args, split="train")
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device=str(args.device), enabled=args.amp)

    _ = train_pirads.train_epoch(model, loader, optimizer, scaler=scaler, epoch=0, args=args)
    _ = train_pirads.val_epoch(model, loader, epoch=0, args=args)

    cspca_model = CSPCAModel(backbone=model).to(args.device)
    optimizer_cspca = torch.optim.AdamW(cspca_model.parameters(), lr=1e-5)
    _ = train_cspca.train_epoch(cspca_model, loader, optimizer_cspca, epoch=0, args=args)
    _ = train_cspca.val_epoch(cspca_model, loader, epoch=0, args=args)
