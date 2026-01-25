import argparse
import collections.abc
import os
import shutil
import time
import yaml
from scipy.stats import pearsonr
import gdown
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp 
import torch.nn as nn
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist, ITKReader, NumpyReader, PersistentDataset
from monai.data.wsi_reader import WSIReader
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
from sklearn.metrics import cohen_kappa_score, roc_curve, confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import default_collate
from torchvision.models.resnet import ResNet50_Weights
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
from AIAH_utility.viewer import BasicViewer
from scipy.special import expit
import nrrd
import random
from sklearn.metrics import roc_auc_score
import SimpleITK as sitk 
from AIAH_utility.viewer import BasicViewer
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import monai

def train_epoch(cspca_model, loader, optimizer, epoch, args):
    cspca_model.train()
    criterion = nn.BCELoss() 
    loss = 0.0
    run_loss = CumulativeAverage()
    TARGETS = Cumulative()
    PREDS = Cumulative()
    
    for idx, batch_data in enumerate(loader):
        data = batch_data["image"].as_subclass(torch.Tensor).to(args.device)
        target = batch_data["label"].as_subclass(torch.Tensor).to(args.device)

        optimizer.zero_grad()
        output = cspca_model(data)
        output = output.squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        TARGETS.extend(target.detach().cpu())
        PREDS.extend(output.detach().cpu())
        run_loss.append(loss.item())

    loss_epoch = run_loss.aggregate()
    target_list = TARGETS.get_buffer().cpu().numpy()
    pred_list = PREDS.get_buffer().cpu().numpy()
    auc_epoch = roc_auc_score(target_list, pred_list)

    return loss_epoch, auc_epoch

def val_epoch(cspca_model, loader, epoch, args):
    cspca_model.eval()
    criterion = nn.BCELoss() 
    loss = 0.0
    run_loss = CumulativeAverage()
    TARGETS = Cumulative()
    PREDS = Cumulative()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data["image"].as_subclass(torch.Tensor).to(args.device)
            target = batch_data["label"].as_subclass(torch.Tensor).to(args.device)

            output = cspca_model(data)
            output = output.squeeze(1)
            loss = criterion(output, target)

            TARGETS.extend(target.detach().cpu())
            PREDS.extend(output.detach().cpu())
            run_loss.append(loss.item())

    loss_epoch = run_loss.aggregate()
    target_list = TARGETS.get_buffer().cpu().numpy()
    pred_list = PREDS.get_buffer().cpu().numpy()
    auc_epoch = roc_auc_score(target_list, pred_list)
    y_pred_categoric = (pred_list >= 0.5)
    tn, fp, fn, tp = confusion_matrix(target_list, y_pred_categoric).ravel()
    sens_epoch = tp / (tp + fn)
    spec_epoch = tn / (tn + fp)
    val_epoch_metric = {'epoch': epoch, 'loss': loss_epoch, 'auc': auc_epoch, 'sensitivity': sens_epoch, 'specificity': spec_epoch}
    return val_epoch_metric

