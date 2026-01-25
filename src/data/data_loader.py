import argparse
import os
import numpy as np
from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist, ITKReader, NumpyReader, PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    ScaleIntensityRanged,
    SplitDimd,
    ToTensord,
    ConcatItemsd, 
    SelectItemsd,
    EnsureChannelFirstd,
    RepeatChanneld,
    DeleteItemsd,
    EnsureTyped,
    ClipIntensityPercentilesd,
    MaskIntensityd,
    RandCropByPosNegLabeld,
    NormalizeIntensityd,
    SqueezeDimd,
    ScaleIntensityd,
    ScaleIntensityd,
    Transposed,
    RandWeightedCropd,
)
from .custom_transforms import (
    NormalizeIntensity_customd, 
    ClipMaskIntensityPercentilesd, 
    ElementwiseProductd,
)
import torch
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
from typing import Literal
import monai
import collections.abc

def list_data_collate(batch: collections.abc.Sequence):
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



def data_transform(args):

    if args.use_heatmap:
        transform = Compose(
            [
                LoadImaged(keys=["image", "mask","dwi", "adc", "heatmap"], reader=ITKReader(), ensure_channel_first=True, dtype=np.float32),
                ClipMaskIntensityPercentilesd(keys=["image"], lower=0, upper=99.5, mask_key="mask"),
                ConcatItemsd(keys=["image", "dwi", "adc"], name="image", dim=0),  # stacks to (3, H, W)
                NormalizeIntensity_customd(keys=["image"], channel_wise=True, mask_key="mask"),
                ElementwiseProductd(keys=["mask", "heatmap"], output_key="final_heatmap"),
                RandWeightedCropd(keys=["image", "final_heatmap"],
                                w_key="final_heatmap",
                                spatial_size=(args.tile_size,args.tile_size,args.depth),
                                num_samples=args.tile_count),
                EnsureTyped(keys=["label"], dtype=torch.float32),
                Transposed(keys=["image"], indices=(0, 3, 1, 2)),
                DeleteItemsd(keys=['mask', 'dwi', 'adc', 'heatmap']),
                ToTensord(keys=["image", "label", "final_heatmap"]),
            ]
        )
    else:
        transform = Compose(
            [
                LoadImaged(keys=["image", "mask","dwi", "adc"], reader=ITKReader(), ensure_channel_first=True, dtype=np.float32),
                ClipMaskIntensityPercentilesd(keys=["image"], lower=0, upper=99.5, mask_key="mask"),
                ConcatItemsd(keys=["image", "dwi", "adc"], name="image", dim=0),  # stacks to (3, H, W)
                NormalizeIntensityd(keys=["image"], channel_wise=True),
                RandCropByPosNegLabeld(keys=["image"],
                                label_key="mask",
                                spatial_size=(args.tile_size,args.tile_size,args.depth),
                                pos=1,
                                neg=0,
                                num_samples=args.tile_count),
                EnsureTyped(keys=["label"], dtype=torch.float32),
                Transposed(keys=["image"], indices=(0, 3, 1, 2)),
                DeleteItemsd(keys=['mask', 'dwi', 'adc']),
                ToTensord(keys=["image", "label"]),
            ]
        )
    return transform

def get_dataloader(args, split: Literal["train", "test"]):

    data_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key=split,
        base_dir=args.data_root,
    )
    if args.dry_run:
        data_list = data_list[:8]  # Use only 8 samples for dry run
    cache_dir_ = os.path.join(args.logdir, "cache")
    os.makedirs(os.path.join(cache_dir_, split), exist_ok=True)
    transform = data_transform(args)
    dataset = PersistentDataset(data=data_list, transform=transform, cache_dir= os.path.join(cache_dir_, split))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context="spawn" if args.workers > 0 else None,
        sampler=None,
        collate_fn=list_data_collate,
    )
    return loader

