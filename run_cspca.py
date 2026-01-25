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

from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import default_collate
from torchvision.models.resnet import ResNet50_Weights
import shutil
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism

import matplotlib.pyplot as plt

import wandb
import math
import logging
from pathlib import Path


from src.model.MIL import MILModel_3D
from src.model.csPCa_model import csPCa_Model
from src.data.data_loader import get_dataloader
from src.utils import save_cspca_checkpoint, get_metrics, setup_logging
from src.train.train_cspca import train_epoch, val_epoch

def main_worker(args):

    mil_model = MILModel_3D(
        num_classes=args.num_classes,  
        mil_mode=args.mil_mode 
    ).to(args.device)
    cache_dir_path = Path(os.path.join(args.logdir, "cache"))

    if args.mode == 'train':

        checkpoint = torch.load(args.checkpoint_pirads, weights_only=False, map_location="cpu")
        mil_model.load_state_dict(checkpoint["state_dict"])
        mil_model = mil_model.to(args.device)
        
        model_dir = os.path.join(args.project_dir,'models')
        metrics_dict = {'auc':[], 'sensitivity':[], 'specificity':[]}
        for st in list(range(args.num_seeds)):
            set_determinism(seed=st)
            
            train_loader = get_dataloader(args, split="train")
            valid_loader = get_dataloader(args, split="test")
            cspca_model = csPCa_Model(backbone=mil_model).to(args.device)
            for submodule in [cspca_model.backbone.net, 
                            cspca_model.backbone.myfc, 
                            cspca_model.backbone.transformer]:
                for param in submodule.parameters():
                    param.requires_grad = False

            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, cspca_model.parameters()), lr=args.optim_lr)

            old_loss = float('inf')
            old_auc = 0.0
            for epoch in range(args.epochs):
                train_loss, train_auc = train_epoch(cspca_model, train_loader, optimizer, epoch=epoch, args=args)
                logging.info(f"STATE {st} EPOCH {epoch} TRAIN loss: {train_loss:.4f} AUC: {train_auc:.4f}")
                val_metric = val_epoch(cspca_model, valid_loader, epoch=epoch, args=args)
                logging.info(f"STATE {st} EPOCH {epoch} VAL loss: {val_metric['loss']:.4f} AUC: {val_metric['auc']:.4f}")
                val_metric['state'] = st
                if val_metric['loss'] < old_loss:
                    old_loss = val_metric['loss']
                    old_auc = val_metric['auc']
                    sensitivity = val_metric['sensitivity']
                    specificity = val_metric['specificity']
                    if len(metrics_dict['auc']) == 0:
                        save_cspca_checkpoint(cspca_model, val_metric, model_dir)
                    elif val_metric['auc'] >= max(metrics_dict['auc']):
                        save_cspca_checkpoint(cspca_model, val_metric, model_dir)

            metrics_dict['auc'].append(old_auc)
            metrics_dict['sensitivity'].append(sensitivity)
            metrics_dict['specificity'].append(specificity)
            if cache_dir_path.exists() and cache_dir_path.is_dir():
                shutil.rmtree(cache_dir_path)

        get_metrics(metrics_dict)

    elif args.mode == 'test':

        cspca_model = csPCa_Model(backbone=mil_model).to(args.device)
        checkpt = torch.load(args.checkpoint_cspca, map_location="cpu")
        cspca_model.load_state_dict(checkpt['state_dict'])
        cspca_model = cspca_model.to(args.device)
        if 'auc' in checkpt and 'sensitivity' in checkpt and 'specificity' in checkpt:
            auc, sens, spec = checkpt['auc'], checkpt['sensitivity'], checkpt['specificity']
            logging.info(f"csPCa Model loaded from {args.checkpoint_cspca} with AUC: {auc}, Sensitivity: {sens}, Specificity: {spec} on the test set.")
        else:
            logging.info(f"csPCa Model loaded from {args.checkpoint_cspca}.")
        
        metrics_dict = {'auc':[], 'sensitivity':[], 'specificity':[]}
        for st in list(range(args.num_seeds)):
            set_determinism(seed=st)
            test_loader = get_dataloader(args, split="test")
            test_metric = val_epoch(cspca_model, test_loader, epoch=0, args=args)
            metrics_dict['auc'].append(test_metric['auc'])
            metrics_dict['sensitivity'].append(test_metric['sensitivity'])
            metrics_dict['specificity'].append(test_metric['specificity'])

            if cache_dir_path.exists() and cache_dir_path.is_dir():
                shutil.rmtree(cache_dir_path)

        get_metrics(metrics_dict)




def parse_args():
    parser = argparse.ArgumentParser(description="Multiple Instance Learning (MIL) for csPCa risk prediction.")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Operation mode: train or infer')
    parser.add_argument('--run_name', type=str, default='train_cspca', help='run name for log file')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument(
        "--project_dir", default=None, help="path to project firectory"
    )
    parser.add_argument(
        "--data_root", default=None, help="path to root folder of images"
    )
    parser.add_argument("--dataset_json", default=None, type=str, help="path to dataset json file")
    parser.add_argument("--num_classes", default=4, type=int, help="number of output classes")
    parser.add_argument("--mil_mode", default="att_trans", help="MIL algorithm: choose either att_trans or att_pyramid")
    parser.add_argument(
        "--tile_count", default=24, type=int, help="number of patches (instances) to extract from MRI input"
    )
    parser.add_argument("--tile_size", default=64, type=int, help="size of square patch (instance) in pixels")
    parser.add_argument("--depth", default=3, type=int, help="number of slices in each 3D patch (instance)")
    parser.add_argument(
        "--use_heatmap", action="store_true",
        help="enable weak attention heatmap guided patch generation"
    )
    parser.add_argument(
        "--no_heatmap", dest="use_heatmap", action="store_false",
        help="disable heatmap"
    )
    parser.set_defaults(use_heatmap=True)
    parser.add_argument("--workers", default=2, type=int, help="number of workers for data loading")
    #parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint_pirads", default=None, help="Load PI-RADS model")
    parser.add_argument("--epochs", "--max_epochs", default=30, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="number of MRI scans per batch")
    parser.add_argument("--optim_lr", default=2e-4, type=float, help="initial learning rate")
    #parser.add_argument("--amp", action="store_true", help="use AMP, recommended")
    parser.add_argument(
        "--val_every",
        "--val_interval",
        default=1,
        type=int,
        help="run validation after this number of epochs, default 1 to run every epoch",
    )
    parser.add_argument("--dry_run", action="store_true", help="Run the script in dry-run mode (default: False)")
    parser.add_argument("--checkpoint_cspca", default=None, help="load existing checkpoint")
    parser.add_argument("--num_seeds", default=20, type=int, help="number of seeds to be run to build CI")
    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
            args.__dict__.update(config)



    return args



if __name__ == "__main__":
    args = parse_args()
    args.logdir = os.path.join(args.project_dir, "logs", args.run_name)
    os.makedirs(args.logdir, exist_ok=True)
    args.logfile = os.path.join(args.logdir, f"{args.run_name}.log")
    setup_logging(args.logfile)


    logging.info("Argument values:")
    for k, v in vars(args).items():
        logging.info(f"{k} => {v}")
    logging.info("-----------------")

    if args.dataset_json is None:
        logging.error('Dataset path not provided. Quitting.')
        sys.exit(1)
    if args.checkpoint_pirads is None and args.mode == 'train':
        logging.error('PI-RADS checkpoint path not provided. Quitting.')
        sys.exit(1)
    elif args.checkpoint_cspca is None and args.mode == 'test':
        logging.error('csPCa checkpoint path not provided. Quitting.')
        sys.exit(1)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True

    if args.dry_run:
        logging.info("Dry run mode enabled.")
        args.epochs = 2
        args.batch_size = 2
        args.workers = 0
        args.num_seeds = 2
        args.wandb = False


    main_worker(args)
