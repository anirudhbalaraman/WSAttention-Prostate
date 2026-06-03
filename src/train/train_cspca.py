import torch
import torch.nn as nn
from monai.metrics import Cumulative, CumulativeAverage
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse


def get_lambda_att(epoch: int, max_lambda: float = 2.0, warmup_epochs: int = 10) -> float:
    if epoch < warmup_epochs:
        return (epoch / warmup_epochs) * max_lambda
    else:
        return max_lambda
    
def get_attention_scores(
    data: torch.Tensor,
    target: torch.Tensor,
    heatmap: torch.Tensor,
    mask: torch.Tensor,
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
        raw_scores_unfil = heatmap_patches.view(len(heatmap_patches), -1).sum(dim=1)

        prostate_mask = mask[i]
        mask_patches = prostate_mask.squeeze(1)
        valid_counts = (mask_patches != 0).sum(dim=(1, 2, 3))

        raw_scores = raw_scores_unfil / valid_counts
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
    att_labels = att_labels**4  # Sharpening
    att_labels = att_labels / att_labels.sum(dim=1, keepdim=True)

    return att_labels, shuffled_images

def train_epoch(cspca_model, loader, optimizer, epoch, args):
    
    lambda_att = get_lambda_att(epoch, warmup_epochs=25)
    
    cspca_model.train()
    criterion = nn.BCEWithLogitsLoss()
    att_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)

    run_att_loss = CumulativeAverage()
    loss = 0.0
    run_loss = CumulativeAverage()
    targets_cumulative = Cumulative()
    preds_cumulative = Cumulative()

    for _, batch_data in enumerate(loader):
        eps = 1e-8
        data = batch_data["image"].as_subclass(torch.Tensor).to(args.device)
        target = batch_data["label"].as_subclass(torch.Tensor).to(args.device)
        psa_data = batch_data["psa"].as_subclass(torch.Tensor).to(args.device)
        
        if args.use_heatmap:
            att_labels, shuffled_images = get_attention_scores(
                data, target, batch_data["final_heatmap"], batch_data["smooth_mask"], args
            )
            att_labels = att_labels + eps
        else:
            shuffled_images = data.to(args.device)
            
        optimizer.zero_grad()
        output = cspca_model(x = shuffled_images, psa_data = psa_data)
        output = output.squeeze(1)
        class_loss = criterion(output, target)
        if args.use_heatmap:
            sh = shuffled_images.shape
            x = shuffled_images.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4], sh[5])
            x = cspca_model.backbone.net(x)
            x = x.reshape(sh[0], sh[1], -1)
            x = x.to(torch.float32)
            x = cspca_model.backbone.transformer(x)
            x_detach = x.detach()
            a = cspca_model.backbone.attention(x_detach)
            a = a.squeeze(-1)
            a = a + eps
            att_preds = torch.softmax(a, dim=1)
            attn_loss = 1 - att_criterion(att_preds, att_labels).mean()
            loss = class_loss + (lambda_att * attn_loss)
        else:
            loss = class_loss
            attn_loss = torch.tensor(0.0)
  
                
        loss.backward()
        optimizer.step()

        targets_cumulative.extend(target.detach().cpu())
        preds_cumulative.extend(output.detach().cpu())
        run_loss.append(loss.item())
        run_att_loss.append(attn_loss.item())

    loss_epoch = run_loss.aggregate()
    attn_loss_epoch = run_att_loss.aggregate()
    target_list = targets_cumulative.get_buffer().cpu().numpy()
    pred_list = preds_cumulative.get_buffer().cpu().numpy()
    auc_epoch = roc_auc_score(target_list, pred_list)

    return loss_epoch, attn_loss_epoch, auc_epoch


def val_epoch(cspca_model, loader, epoch, args):
    cspca_model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loss = 0.0
    run_loss = CumulativeAverage()
    targets_cumulative = Cumulative()
    preds_cumulative = Cumulative()
    with torch.no_grad():
        for _, batch_data in enumerate(loader):
            data = batch_data["image"].as_subclass(torch.Tensor).to(args.device)
            target = batch_data["label"].as_subclass(torch.Tensor).to(args.device)
            psa_data = batch_data["psa"].as_subclass(torch.Tensor).to(args.device)

            output = cspca_model(x = data, psa_data = psa_data)
            output = output.squeeze(1)
            loss = criterion(output, target)

            targets_cumulative.extend(target.detach().cpu())
            preds_cumulative.extend(output.detach().cpu())
            run_loss.append(loss.item())

    loss_epoch = run_loss.aggregate()
    target_list = targets_cumulative.get_buffer().cpu().numpy()
    pred_list = preds_cumulative.get_buffer().cpu().numpy()
    auc_epoch = roc_auc_score(target_list, pred_list)
    y_pred_categoric = pred_list >= 0.5
    tn, fp, fn, tp = confusion_matrix(target_list, y_pred_categoric).ravel()
    sens_epoch = tp / (tp + fn)
    spec_epoch = tn / (tn + fp)
    val_epoch_metric = {
        "epoch": epoch,
        "loss": loss_epoch,
        "auc": auc_epoch,
        "sensitivity": sens_epoch,
        "specificity": spec_epoch,
    }
    return val_epoch_metric
