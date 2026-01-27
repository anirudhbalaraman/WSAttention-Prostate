import argparse
import os
import shutil
import time
import yaml
import sys
import gdown
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.metrics import Cumulative, CumulativeAverage
from monai.networks.nets import milmodel, resnet, MILModel
from monai.transforms import (
    Compose,
    GridPatchd,
    LoadImaged,
    MapTransform,
    RandFlipd,
    RandGridPatchd,
    RandRotate90d,
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
    HistogramNormalized,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
    NormalizeIntensityd,
    SqueezeDimd,
    CropForegroundd,
    ScaleIntensityd,
    SpatialPadd,
    CenterSpatialCropd,
    ScaleIntensityd,
    Transposed,
    RandWeightedCropd,
)
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import default_collate
from torchvision.models.resnet import ResNet50_Weights
from .data.custom_transforms import ClipMaskIntensityPercentilesd, NormalizeIntensity_customd
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.patches as patches

import matplotlib.pyplot as plt

import wandb
import math
from monai.data import Dataset, load_decathlon_datalist, ITKReader, NumpyReader, PersistentDataset

from src.model.MIL import MILModel_3D
from src.model.csPCa_model import csPCa_Model

import logging
from pathlib import Path

def save_pirads_checkpoint(model, epoch, args, filename="model.pth", best_acc=0):

    """Save checkpoint"""

    state_dict = model.state_dict()

    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}

    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    logging.info("Saving checkpoint", filename)

def save_cspca_checkpoint(model, val_metric, model_dir):
    state_dict = model.state_dict()
    save_dict = {
        'epoch' : val_metric['epoch'],
        'loss' : val_metric['loss'],
        'auc' : val_metric['auc'],
        'sensitivity' : val_metric['sensitivity'],
        'specificity' : val_metric['specificity'],
        'state' : val_metric['state'],
        'state_dict' : state_dict,
    }
    torch.save(save_dict, os.path.join(model_dir,f"cspca_model.pth"))
    logging.info('Saving model with auc: ', str(val_metric['auc']))

def get_metrics(metric_dict: dict):
    for metric_name, metric_list in metric_dict.items():
        metric_list = np.array(metric_list)
        lower = np.percentile(metric_list, 2.5)
        upper = np.percentile(metric_list, 97.5)
        mean_metric = np.mean(metric_list)
        logging.info(f"Mean {metric_name}: {mean_metric:.3f}")
        logging.info(f"95% CI: ({lower:.3f}, {upper:.3f})")

def setup_logging(log_file):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.write_text("")  # overwrite with empty string
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ],
    )

def validate_steps(steps):
    REQUIRES = {
        "get_segmentation_mask": ["register_and_crop"],
        "histogram_match": ["get_segmentation_mask", "register_and_crop"],
        "get_heatmap": ["get_segmentation_mask", "histogram_match", "register_and_crop"],
    }
    for i, step in enumerate(steps):
        required = REQUIRES.get(step, [])
        for req in required:
            if req not in steps[:i]:
                logging.error(
                    f"Step '{step}' requires '{req}' to be executed before it. "
                    f"Given order: {steps}"
                )
                sys.exit(1)

def get_patch_coordinate(patches_top_5, parent_image, args):

    sample = np.array([i.transpose(1,2,0) for i in patches_top_5])
    coords = []
    rows, h, w, slices = sample.shape

    for i in range(rows):
        for j in range(slices):
            if j == 0:
                for k in range(parent_image.shape[2]):
                    img_temp = parent_image[:, :, k]
                    H, W = img_temp.shape
                    h, w = sample[i, :, :, j].shape
                    a,b = 0, 0  # Initialize a and b
                    bool1 = False
                    for l in range(H - h + 1):
                        for m in range(W - w + 1):
                            if np.array_equal(img_temp[l:l+h, m:m+w], sample[i, :, :, j]):
                                a,b = l, m  # top-left corner
                                coords.append((a,b,k))
                                bool1 = True
                                break
                        if bool1:
                            break
                        
                    if bool1:
                        break

    return coords


def get_parent_image(args):
    transform_image = Compose(
        [
            LoadImaged(keys=["image", "mask"], reader=ITKReader(), ensure_channel_first=True, dtype=np.float32),
            ClipMaskIntensityPercentilesd(keys=["image"], lower=0, upper=99.5, mask_key="mask"),
            NormalizeIntensity_customd(keys=["image"], mask_key="mask", channel_wise=True),
            EnsureTyped(keys=["label"], dtype=torch.float32),
            ToTensord(keys=["image", "label"]),
        ]
    )
    dataset_image = Dataset(data=args.data_list, transform=transform_image)
    return dataset_image[0]['image'][0].numpy()

'''
def visualise_patches():
    sample = np.array([i.transpose(1,2,0) for i in patches_top_5])
    rows = len(patches_top_5)
    img = sample[0]
    coords = []
    rows, h, w, slices = sample.shape

    fig, axes = plt.subplots(nrows=rows, ncols=slices, figsize=(slices * 3, rows * 3))

    for i in range(rows):
        for j in range(slices):
            ax = axes[i, j]
            
            if j == 0:
            
                for k in range(parent_image.shape[2]):
                    img_temp = parent_image[:, :, k]
                    H, W = img_temp.shape
                    h, w = sample[i, :, :, j].shape
                    a,b = 0, 0  # Initialize a and b
                    bool1 = False
                    for l in range(H - h + 1):
                        for m in range(W - w + 1):
                            if np.array_equal(img_temp[l:l+h, m:m+w], sample[i, :, :, j]):
                                a,b = l, m  # top-left corner
                                coords.append((a,b,k))
                                bool1 = True
                                break
                        if bool1:
                            break
                        
                    if bool1:
                        break

                


            ax.imshow(parent_image[:, :, k+j], cmap='gray')
            rect = patches.Rectangle((b, a), args.tile_size, args.tile_size,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.axis('off')
        

    plt.tight_layout()
    plt.show()
    a=1
'''