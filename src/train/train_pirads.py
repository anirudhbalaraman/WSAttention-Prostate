import argparse
import collections.abc
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
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import wandb
import math
import logging
from pathlib import Path
from src.data.data_loader import get_dataloader
from src.utils import save_pirads_checkpoint, setup_logging

def get_lambda_att(epoch, max_lambda=2.0, warmup_epochs=10):
    if epoch < warmup_epochs:
        return (epoch / warmup_epochs) * max_lambda
    else:
        return max_lambda

def get_attention_scores(data, target, heatmap, args):
    attention_score = torch.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        sample = heatmap[i]
        heatmap_patches = sample.squeeze(1)
        raw_scores = heatmap_patches.view(len(heatmap_patches), -1).sum(dim=1)  
        attention_score[i] = raw_scores / raw_scores.sum() 
    shuffled_images = torch.empty_like(data).to(args.device)
    att_labels = torch.empty_like(attention_score).to(args.device)
    for i in range(data.shape[0]):  
        perm = torch.randperm(data.shape[1])  
        shuffled_images[i] = data[i, perm]
        att_labels[i] = attention_score[i, perm] 
        
    att_labels[torch.argwhere(target < 1)] = torch.ones_like(att_labels[0]) / len(att_labels[0])# Setting attention scores for cases
    att_labels = att_labels ** 2  # Sharpening
    att_labels = att_labels / att_labels.sum(dim=1, keepdim=True)
    
    return att_labels, shuffled_images

def train_epoch(model, loader, optimizer, scaler, epoch, args):
    """One train epoch over the dataset"""
    lambda_att = get_lambda_att(epoch, warmup_epochs=25)

    model.train()
    criterion = nn.CrossEntropyLoss()
    att_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)

    run_att_loss = CumulativeAverage()
    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()
    batch_norm = CumulativeAverage()

    start_time = time.time()
    loss, acc = 0.0, 0.0

    for idx, batch_data in enumerate(loader):

        eps = 1e-8
        data = batch_data["image"].as_subclass(torch.Tensor)
        target = batch_data["label"].as_subclass(torch.Tensor).to(args.device)
        target = target.long()
        if args.use_heatmap:
            att_labels, shuffled_images = get_attention_scores(data, target, batch_data['final_heatmap'], args)
            att_labels = att_labels + eps
        else:
            shuffled_images = data.to(args.device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=str(args.device), enabled=args.amp):
            # Classification Loss
            logits_attn = model(shuffled_images, no_head=True)
            x = logits_attn.to(torch.float32)
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)
            a = model.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            logits = model.myfc(x)
            class_loss = criterion(logits, target)
            # Attention Loss
            if args.use_heatmap:
                y = logits_attn.to(torch.float32)
                y = y.permute(1, 0, 2)
                y = model.transformer(y)
                y_detach = y.permute(1, 0, 2).detach()
                b = model.attention(y_detach)
                b = b.squeeze(-1)
                b = b + eps
                att_preds = torch.softmax(b, dim=1)
                attn_loss = 1 - att_criterion(att_preds, att_labels).mean()
                loss = class_loss + (lambda_att*attn_loss)
            else:
                loss = class_loss
                attn_loss = torch.tensor(0.0)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        if not torch.isfinite(total_norm):
            logging.warning("Non-finite gradient norm detected, skipping batch.")
            optimizer.zero_grad()
        else:
            scaler.step(optimizer)
            scaler.update()
            shuffled_images = shuffled_images.to('cpu')
            logits = logits.to('cpu')
            logits_attn = logits_attn.to('cpu')
            target = target.to('cpu')
            
            batch_norm.append(total_norm)
            pred = torch.argmax(logits, dim=1)
            acc = (pred == target).sum() / len(pred)

            run_att_loss.append(attn_loss.detach().cpu())
            run_loss.append(loss.detach().cpu())
            run_acc.append(acc.detach().cpu())
            logging.info(
                "Epoch {}/{} {}/{} loss: {:.4f} attention loss: {:.4f} acc: {:.4f} grad norm: {:.4f} time {:.2f}s".format(
                    epoch,
                    args.epochs,
                    idx,
                    len(loader),
                    loss,
                    attn_loss,
                    acc,
                    total_norm,
                    time.time() - start_time
                )
            )
            start_time = time.time()
    
    del data, target, shuffled_images, logits, logits_attn
    torch.cuda.empty_cache()
    batch_norm_epoch = batch_norm.aggregate()
    attn_loss_epoch = run_att_loss.aggregate()
    loss_epoch = run_loss.aggregate()
    acc_epoch = run_acc.aggregate()
    return loss_epoch, acc_epoch, attn_loss_epoch, batch_norm_epoch



def val_epoch(model, loader, epoch, args):

    criterion = nn.CrossEntropyLoss()

    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()
    PREDS = Cumulative()
    TARGETS = Cumulative()

    start_time = time.time()
    loss, acc = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):

            data = batch_data["image"].as_subclass(torch.Tensor).to(args.device)
            target = batch_data["label"].as_subclass(torch.Tensor).to(args.device)
            target = target.long()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(data)
                loss = criterion(logits, target)

            data = data.to('cpu')
            target = target.to('cpu')
            logits = logits.to('cpu')
            pred = torch.argmax(logits, dim=1)
            acc = (pred == target).sum() / len(target)

            run_loss.append(loss.detach().cpu())
            run_acc.append(acc.detach().cpu())
            PREDS.extend(pred.detach().cpu())
            TARGETS.extend(target.detach().cpu())
            logging.info(
                "Val epoch {}/{} {}/{} loss: {:.4f} acc: {:.4f} time {:.2f}s".format(
                    epoch, args.epochs, idx, len(loader), loss, acc, time.time() - start_time
                )
            )
            start_time = time.time()
            
            del data, target, logits
            torch.cuda.empty_cache()

        # Calculate QWK metric (Quadratic Weigted Kappa) https://en.wikipedia.org/wiki/Cohen%27s_kappa
        PREDS = PREDS.get_buffer().cpu().numpy()
        TARGETS = TARGETS.get_buffer().cpu().numpy()
        loss_epoch = run_loss.aggregate()
        acc_epoch = run_acc.aggregate()
        qwk = cohen_kappa_score(TARGETS.astype(np.float64),PREDS.astype(np.float64))
    return loss_epoch, acc_epoch, qwk





