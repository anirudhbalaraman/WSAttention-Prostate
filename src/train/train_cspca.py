import torch
import torch.nn as nn
from monai.metrics import Cumulative, CumulativeAverage
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


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
