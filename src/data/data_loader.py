import argparse
import os
from typing import Literal

import numpy as np
import torch
from monai.data import PersistentDataset, load_decathlon_datalist
from monai.transforms import (
    Compose,
    ConcatItemsd,
    DeleteItemsd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandWeightedCropd,
    ToTensord,
    Transform,
    Transposed,
)
from torch.utils.data.dataloader import default_collate

from .custom_transforms import (
    ClipMaskIntensityPercentilesd,
    ElementwiseProductd,
    NormalizeIntensity_customd,
)

class DummyMILDataset(torch.utils.data.Dataset):
    def __init__(self, args, num_samples=8):
        self.num_samples = num_samples
        self.args = args

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Simulate the output of your 'data_transform'
        # A list of dictionaries, one for each 'tile_count' (patch)
        bag = []
        label_value = float(index % 2)
        for _ in range(self.args.tile_count):
            item = {
                # Shape: (Channels=3, Depth, H, W) based on your Transposed(indices=(0, 3, 1, 2))
                "image": torch.randn(3, self.args.depth, self.args.tile_size, self.args.tile_size),
                "label": torch.tensor(label_value, dtype=torch.float32)
            }
            if self.args.use_heatmap:
                item["final_heatmap"] = torch.randn(1, self.args.depth, self.args.tile_size, self.args.tile_size)
            bag.append(item)
        return bag

def list_data_collate(batch: list):
    """
    Combine instances from a list of dicts into a single dict, by stacking them along first dim
    [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
    followed by the default collate which will form a batch BxNx3xHxW
    """

    for i, item in enumerate(batch):
        data = item[0]
        data["image"] = torch.stack([ix["image"] for ix in item], dim=0)

        if all("final_heatmap" in ix for ix in item):
            data["final_heatmap"] = torch.stack([ix["final_heatmap"] for ix in item], dim=0)

        batch[i] = data
    return default_collate(batch)


def data_transform(args: argparse.Namespace) -> Transform:
    if args.use_heatmap:
        transform = Compose(
            [
                LoadImaged(
                    keys=["image", "mask", "dwi", "adc", "heatmap"],
                    reader="ITKReader",
                    ensure_channel_first=True,
                    dtype=np.float32,
                ),
                ClipMaskIntensityPercentilesd(keys=["image"], lower=0, upper=99.5, mask_key="mask"),
                ConcatItemsd(
                    keys=["image", "dwi", "adc"], name="image", dim=0
                ),  # stacks to (3, H, W)
                NormalizeIntensity_customd(keys=["image"], channel_wise=True, mask_key="mask"),
                ElementwiseProductd(keys=["mask", "heatmap"], output_key="final_heatmap"),
                RandWeightedCropd(
                    keys=["image", "final_heatmap"],
                    w_key="final_heatmap",
                    spatial_size=(args.tile_size, args.tile_size, args.depth),
                    num_samples=args.tile_count,
                ),
                EnsureTyped(keys=["label"], dtype=torch.float32),
                Transposed(keys=["image"], indices=(0, 3, 1, 2)),
                DeleteItemsd(keys=["mask", "dwi", "adc", "heatmap"]),
                ToTensord(keys=["image", "label", "final_heatmap"]),
            ]
        )
    else:
        transform = Compose(
            [
                LoadImaged(
                    keys=["image", "mask", "dwi", "adc"],
                    reader="ITKReader",
                    ensure_channel_first=True,
                    dtype=np.float32,
                ),
                ClipMaskIntensityPercentilesd(keys=["image"], lower=0, upper=99.5, mask_key="mask"),
                ConcatItemsd(
                    keys=["image", "dwi", "adc"], name="image", dim=0
                ),  # stacks to (3, H, W)
                NormalizeIntensityd(keys=["image"], channel_wise=True),
                RandCropByPosNegLabeld(
                    keys=["image"],
                    label_key="mask",
                    spatial_size=(args.tile_size, args.tile_size, args.depth),
                    pos=1,
                    neg=0,
                    num_samples=args.tile_count,
                ),
                EnsureTyped(keys=["label"], dtype=torch.float32),
                Transposed(keys=["image"], indices=(0, 3, 1, 2)),
                DeleteItemsd(keys=["mask", "dwi", "adc"]),
                ToTensord(keys=["image", "label"]),
            ]
        )
    return transform


def get_dataloader(
    args: argparse.Namespace, split: Literal["train", "test"]
) -> torch.utils.data.DataLoader:

    if args.dry_run:
        print(f"🛠️  DRY RUN: Creating synthetic {split} dataloader...")
        dummy_ds = DummyMILDataset(args, num_samples=args.batch_size * 2)
        return torch.utils.data.DataLoader(
            dummy_ds,
            batch_size=args.batch_size,
            collate_fn=list_data_collate, # Uses your custom stacking logic
            num_workers=0 # Keep it simple for dry run
        )


    data_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key=split,
        base_dir=args.data_root,
    )
    cache_dir_ = os.path.join(args.logdir, "cache")
    os.makedirs(os.path.join(cache_dir_, split), exist_ok=True)
    transform = data_transform(args)
    dataset = PersistentDataset(
        data=data_list, transform=transform, cache_dir=os.path.join(cache_dir_, split)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context="fork" if args.workers > 0 else None,
        sampler=None,
        collate_fn=list_data_collate,
    )
    return loader
