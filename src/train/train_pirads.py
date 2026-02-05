import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from monai.metrics import Cumulative, CumulativeAverage
from sklearn.metrics import cohen_kappa_score


def get_lambda_att(epoch: int, max_lambda: float = 2.0, warmup_epochs: int = 10) -> float:
    if epoch < warmup_epochs:
        return (epoch / warmup_epochs) * max_lambda
    else:
        return max_lambda


def get_attention_scores(
    data: torch.Tensor,
    target: torch.Tensor,
    heatmap: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention scores from heatmaps and shuffle data accordingly.
    This function generates attention scores based on spatial heatmaps, applies
    sharpening, and creates shuffled versions of the input data and attention
    labels. For PI-RADS 2 (target < 1), uniform attention scores are assigned.
    Args:
        data (torch.Tensor): Input data tensor of shape (batch_size, num_patches, ...).
        target (torch.Tensor): Target labels tensor of shape (batch_size,).
        heatmap (torch.Tensor): Attention heatmap tensor corresponding to input patches.
        args: Arguments object containing device specification.
    Returns:
        tuple: A tuple containing:
            - att_labels (torch.Tensor): Sharpened and normalized attention scores
              of shape (batch_size, num_patches), moved to args.device.
            - shuffled_images (torch.Tensor): Randomly permuted data samples
              of shape (batch_size, num_patches, ...), moved to args.device.
    Note:
        - Attention scores are computed by summing heatmap values across spatial dimensions.
        - Data and attention labels are shuffled with the same permutation per sample.
        - PI-RADS 2 samples receive uniform attention distribution.
        - Attention scores are squared for sharpening and then normalized.
    """

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

    att_labels[torch.argwhere(target < 1)] = torch.ones_like(att_labels[0]) / len(
        att_labels[0]
    )  # For PI-RADS 2, uniform scores across patches
    att_labels = att_labels**2  # Sharpening
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
            att_labels, shuffled_images = get_attention_scores(
                data, target, batch_data["final_heatmap"], args
            )
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
                loss = class_loss + (lambda_att * attn_loss)
            else:
                loss = class_loss
                attn_loss = torch.tensor(0.0)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        if not torch.isfinite(total_norm):
            logging.warning("Non-finite gradient norm detected, skipping batch.")
            optimizer.zero_grad()
            scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()
            batch_norm.append(total_norm)
            pred = torch.argmax(logits, dim=1)
            acc = (pred == target).sum() / len(pred)

            run_att_loss.append(attn_loss.detach().cpu())
            run_loss.append(loss.detach().cpu())
            run_acc.append(acc.detach().cpu())
            logging.info(
                f"Epoch {epoch}/{args.epochs} {idx}/{len(loader)} loss: {loss.item():.4f} attention loss: {attn_loss.item():.4f} acc: {acc:.4f} grad norm: {total_norm:.4f} time {time.time() - start_time:.2f}s"
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
    preds_cumulative = Cumulative()
    targets_cumulative = Cumulative()

    start_time = time.time()
    loss, acc = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data["image"].as_subclass(torch.Tensor).to(args.device)
            target = batch_data["label"].as_subclass(torch.Tensor).to(args.device)
            target = target.long()

            with torch.amp.autocast(device_type=str(args.device), enabled=args.amp):
                logits = model(data)
                loss = criterion(logits, target)

            data = data.to("cpu")
            target = target.to("cpu")
            logits = logits.to("cpu")
            pred = torch.argmax(logits, dim=1)
            acc = (pred == target).sum() / len(target)

            run_loss.append(loss.detach().cpu())
            run_acc.append(acc.detach().cpu())
            preds_cumulative.extend(pred.detach().cpu())
            targets_cumulative.extend(target.detach().cpu())
            logging.info(
                f"Val epoch {epoch}/{args.epochs} {idx}/{len(loader)} loss: {loss:.4f} acc: {acc:.4f} time {time.time() - start_time:.2f}s"
            )
            start_time = time.time()

            del data, target, logits
            torch.cuda.empty_cache()

        # Calculate QWK metric (Quadratic Weigted Kappa) https://en.wikipedia.org/wiki/Cohen%27s_kappa
        preds_cumulative = preds_cumulative.get_buffer().cpu().numpy()
        targets_cumulative = targets_cumulative.get_buffer().cpu().numpy()
        loss_epoch = run_loss.aggregate()
        acc_epoch = run_acc.aggregate()
        qwk = cohen_kappa_score(
            targets_cumulative.astype(np.float64), preds_cumulative.astype(np.float64)
        )
    return loss_epoch, acc_epoch, qwk
