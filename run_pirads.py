import argparse
import os
import shutil
import time
import yaml
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
import wandb
import logging
from pathlib import Path
from src.data.data_loader import get_dataloader
from src.train.train_pirads import train_epoch, val_epoch
from src.model.MIL import MILModel_3D
from src.utils import save_pirads_checkpoint, setup_logging


def main_worker(args):
    if args.device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True

    model = MILModel_3D(num_classes=args.num_classes, mil_mode=args.mil_mode)
    start_epoch = 0
    best_acc = 0.0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        logging.info(
            "=> loaded checkpoint %s (epoch %d) (bestacc %f)",
            args.checkpoint,
            start_epoch,
            best_acc,
        )
    cache_dir_ = os.path.join(args.logdir, "cache")
    model.to(args.device)
    params = model.parameters()
    if args.mode == "train":
        train_loader = get_dataloader(args, split="train")
        valid_loader = get_dataloader(args, split="test")
        logging.info(
            f"Dataset training: {len(train_loader.dataset)}, test: {len(valid_loader.dataset)}"
        )

        if args.mil_mode in ["att_trans", "att_trans_pyramid"]:
            params = [
                {
                    "params": list(model.attention.parameters())
                    + list(model.myfc.parameters())
                    + list(model.net.parameters())
                },
                {"params": list(model.transformer.parameters()), "lr": 6e-5, "weight_decay": 0.1},
            ]

        optimizer = torch.optim.AdamW(params, lr=args.optim_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0
        )
        scaler = torch.amp.GradScaler(device=str(args.device), enabled=args.amp)

        if args.logdir is not None:
            writer = SummaryWriter(log_dir=args.logdir)
            logging.info(f"Writing Tensorboard logs to {writer.log_dir}")
        else:
            writer = None

        # RUN TRAINING
        n_epochs = args.epochs
        val_loss_min = float("inf")
        epochs_no_improve = 0
        for epoch in range(start_epoch, n_epochs):
            logging.info(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            train_loss, train_acc, train_att_loss, batch_norm = train_epoch(
                model, train_loader, optimizer, scaler=scaler, epoch=epoch, args=args
            )
            logging.info(
                "Final training %d/%d loss: %.4f attention loss: %.4f acc: %.4f time %.2fs",
                epoch,
                n_epochs - 1,
                train_loss,
                train_att_loss,
                train_acc,
                time.time() - epoch_time,
            )

            if writer is not None:
                writer.add_scalar("train_loss", train_loss, epoch)
                writer.add_scalar("train_attention_loss", train_att_loss, epoch)
                writer.add_scalar("train_acc", train_acc, epoch)
            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Train Attention Loss": train_att_loss,
                    "Batch Norm": batch_norm,
                },
                step=epoch,
            )

            model_new_best = False
            val_acc = 0
            if (epoch + 1) % args.val_every == 0:
                epoch_time = time.time()
                val_loss, val_acc, qwk = val_epoch(model, valid_loader, epoch=epoch, args=args)

                logging.info(
                    "Final test %d/%d loss: %.4f acc: %.4f qwk: %.4f time %.2fs",
                    epoch,
                    n_epochs - 1,
                    val_loss,
                    val_acc,
                    qwk,
                    time.time() - epoch_time,
                )
                if writer is not None:
                    writer.add_scalar("test_loss", val_loss, epoch)
                    writer.add_scalar("test_acc", val_acc, epoch)
                    writer.add_scalar("test_qwk", qwk, epoch)

                    # val_acc = qwk
                wandb.log(
                    {"Test Loss": val_loss, "Test Accuracy": val_acc, "Cohen Kappa": qwk},
                    step=epoch,
                )
                if val_loss < val_loss_min:
                    logging.info("Loss (%.6f --> %.6f)", val_loss_min, val_loss)
                    val_loss_min = val_loss
                    model_new_best = True

            if args.logdir is not None:
                save_pirads_checkpoint(
                    model, epoch, args, best_acc=val_acc, filename=f"model_{epoch}.pt"
                )
                if model_new_best:
                    logging.info("Copying to model.pt new best model")
                    shutil.copyfile(
                        os.path.join(args.logdir, f"model_{epoch}.pt"),
                        os.path.join(args.logdir, "model.pt"),
                    )
                    epochs_no_improve = 0

                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == args.early_stop:
                        logging.info("Early stopping!")
                        break

            scheduler.step()

        logging.info("ALL DONE")

    elif args.mode == "test":
        kappa_list = []
        for seed in list(range(args.num_seeds)):
            set_determinism(seed=seed)
            valid_loader = get_dataloader(args, split=args.mode)
            logging.info("test:", str(len(valid_loader.dataset)))
            val_loss, val_acc, qwk = val_epoch(model, valid_loader, epoch=0, args=args)
            kappa_list.append(qwk)
            logging.info(f"Seed {seed}, QWK: {qwk}")
            if os.path.exists(cache_dir_):
                logging.info(f"Removing cache directory {cache_dir_}")
                shutil.rmtree(cache_dir_)

        logging.info(f"Mean QWK over {args.num_seeds} seeds: {np.mean(kappa_list)}")

    if os.path.exists(cache_dir_):
        logging.info(f"Removing cache directory {cache_dir_}")
        shutil.rmtree(cache_dir_)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multiple Instance Learning (MIL) for PIRADS Classification."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        required=True,
        help="operation mode: train or infer",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Add this flag to enable WandB logging"
    )
    parser.add_argument(
        "--project_name", type=str, default="Classification_prostate", help="WandB project name"
    )
    parser.add_argument(
        "--run_name", type=str, default="train_pirads", help="run name for WandB logging"
    )
    parser.add_argument("--config", type=str, help="path to YAML config file")
    parser.add_argument("--project_dir", default=None, help="path to project firectory")
    parser.add_argument("--data_root", default=None, help="path to root folder of images")
    parser.add_argument("--dataset_json", default=None, type=str, help="path to dataset json file")
    parser.add_argument("--num_classes", default=4, type=int, help="number of output classes")
    parser.add_argument(
        "--mil_mode",
        default="att_trans",
        help="MIL algorithm: choose either att_trans or att_pyramid",
    )
    parser.add_argument(
        "--tile_count",
        default=24,
        type=int,
        help="number of patches (instances) to extract from MRI input",
    )
    parser.add_argument(
        "--tile_size", default=64, type=int, help="size of square patch (instance) in pixels"
    )
    parser.add_argument(
        "--depth", default=3, type=int, help="number of slices in each 3D patch (instance)"
    )
    parser.add_argument(
        "--use_heatmap",
        action="store_true",
        help="enable weak attention heatmap guided patch generation",
    )
    parser.add_argument(
        "--no_heatmap", dest="use_heatmap", action="store_false", help="disable heatmap"
    )
    parser.set_defaults(use_heatmap=True)
    parser.add_argument("--workers", default=2, type=int, help="number of workers for data loading")

    parser.add_argument("--checkpoint", default=None, help="load existing checkpoint")
    parser.add_argument(
        "--epochs", "--max_epochs", default=50, type=int, help="number of training epochs"
    )
    parser.add_argument("--early_stop", default=40, type=int, help="early stopping criteria")
    parser.add_argument("--batch_size", default=4, type=int, help="number of MRI scans per batch")
    parser.add_argument("--optim_lr", default=3e-5, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=0, type=float, help="optimizer weight decay")
    parser.add_argument("--amp", action="store_true", help="use AMP, recommended")
    parser.add_argument(
        "--val_every",
        "--val_interval",
        default=1,
        type=int,
        help="run validation after this number of epochs, default 1 to run every epoch",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Run the script in dry-run mode (default: False)"
    )
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
            args.__dict__.update(config)
    return args


if __name__ == "__main__":

    args = parse_args()
    if args.project_dir is None:
        args.project_dir = Path(__file__).resolve().parent # Set project directory

    slurm_job_name = os.getenv('SLURM_JOB_NAME') # If the script is submitted via slurm, job name is the run name
    if slurm_job_name:
        args.run_name = slurm_job_name

    args.logdir = os.path.join(args.project_dir, "logs", args.run_name)
    os.makedirs(args.logdir, exist_ok=True)
    args.logfile = os.path.join(args.logdir, f"{args.run_name}.log")
    setup_logging(args.logfile)

    logging.info("Argument values:")
    for k, v in vars(args).items():
        logging.info(f"{k} => {v}")
    logging.info("-----------------")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == torch.device("cpu"):
        args.amp = False
    if args.dataset_json is None:
        logging.error("Dataset JSON file not provided. Quitting.")
        sys.exit(1)
    if args.checkpoint is None and args.mode == "test":
        logging.error("Model checkpoint path not provided. Quitting.")
        sys.exit(1)

    if args.dry_run:
        logging.info("Dry run mode enabled.")
        args.epochs = 2
        args.batch_size = 2
        args.workers = 0
        args.num_seeds = 2
        args.wandb = False

    mode_wandb = "online" if args.wandb and args.mode != "test" else "disabled"

    config_wandb = {
        "learning_rate": args.optim_lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patch size": args.tile_size,
        "patch count": args.tile_count,
    }
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        dir=os.path.join(args.logdir, "wandb"),
        config=config_wandb,
        mode=mode_wandb,
    )

    main_worker(args)

    wandb.finish()
