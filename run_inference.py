import argparse
import os
import yaml
import torch
import logging
from src.model.MIL import MILModel_3D
from src.model.csPCa_model import csPCa_Model
from src.utils import setup_logging, get_parent_image, get_patch_coordinate
from src.preprocessing.register_and_crop import register_files
from src.preprocessing.prostate_mask import get_segmask
from src.preprocessing.histogram_match import histmatch
from src.preprocessing.generate_heatmap import get_heatmap
from src.data.data_loader import data_transform, list_data_collate
from monai.data import Dataset
import json


def parse_args():
    parser = argparse.ArgumentParser(description="File preprocessing")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--t2_dir", default=None, help="Path to T2W files")
    parser.add_argument("--dwi_dir", default=None, help="Path to DWI files")
    parser.add_argument("--adc_dir", default=None, help="Path to ADC files")
    parser.add_argument("--seg_dir", default=None, help="Path to segmentation masks")
    parser.add_argument("--output_dir", default=None, help="Path to output folder")
    parser.add_argument(
        "--margin", default=0.2, type=float, help="Margin to center crop the images"
    )
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--mil_mode", default="att_trans", type=str)
    parser.add_argument("--use_heatmap", default=True, type=bool)
    parser.add_argument("--tile_size", default=64, type=int)
    parser.add_argument("--tile_count", default=24, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--project_dir", default=None, help="Project directory")

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
            args.__dict__.update(config)
    return args


if __name__ == "__main__":
    args = parse_args()
    FUNCTIONS = {
        "register_and_crop": register_files,
        "histogram_match": histmatch,
        "get_segmentation_mask": get_segmask,
        "get_heatmap": get_heatmap,
    }

    args.logfile = os.path.join(args.output_dir, "inference.log")
    setup_logging(args.logfile)
    logging.info("Starting preprocessing")
    steps = ["register_and_crop", "get_segmentation_mask", "histogram_match", "get_heatmap"]
    for step in steps:
        func = FUNCTIONS[step]
        args = func(args)

    logging.info("Preprocessing completed.")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading PIRADS model")
    pirads_model = MILModel_3D(num_classes=args.num_classes, mil_mode=args.mil_mode)
    pirads_checkpoint = torch.load(
        os.path.join(args.project_dir, "models", "pirads.pt"), map_location="cpu"
    )
    pirads_model.load_state_dict(pirads_checkpoint["state_dict"])
    pirads_model.to(args.device)
    logging.info("Loading csPCa model")
    cspca_model = csPCa_Model(backbone=pirads_model).to(args.device)
    checkpt = torch.load(
        os.path.join(args.project_dir, "models", "cspca_model.pth"), map_location="cpu"
    )
    cspca_model.load_state_dict(checkpt["state_dict"])
    cspca_model = cspca_model.to(args.device)

    transform = data_transform(args)
    files = os.listdir(args.t2_dir)
    args.data_list = []
    for file in files:
        temp = {}
        temp["image"] = os.path.join(args.t2_dir, file)
        temp["dwi"] = os.path.join(args.dwi_dir, file)
        temp["adc"] = os.path.join(args.adc_dir, file)
        temp["heatmap"] = os.path.join(args.heatmapdir, file)
        temp["mask"] = os.path.join(args.seg_dir, file)
        temp["label"] = 0  # dummy label
        args.data_list.append(temp)

    dataset = Dataset(data=args.data_list, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        multiprocessing_context=None,
        sampler=None,
        collate_fn=list_data_collate,
    )

    pirads_list = []
    pirads_model.eval()
    cspca_risk_list = []
    cspca_model.eval()
    patches_top_5_list = []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data["image"].as_subclass(torch.Tensor).to(args.device)
            logits = pirads_model(data)
            pirads_score = torch.argmax(logits, dim=1)
            pirads_list.append(pirads_score.item())

            output = cspca_model(data)
            output = output.squeeze(1)
            cspca_risk_list.append(output.item())

            sh = data.shape
            x = data.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4], sh[5])
            x = cspca_model.backbone.net(x)
            x = x.reshape(sh[0], sh[1], -1)
            x = x.permute(1, 0, 2)
            x = cspca_model.backbone.transformer(x)
            x = x.permute(1, 0, 2)
            a = cspca_model.backbone.attention(x)
            a = torch.softmax(a, dim=1)
            a = a.view(-1)
            top5_values, top5_indices = torch.topk(a, 5)

            patches_top_5 = []
            for i in range(5):
                patch_temp = data[0, top5_indices.cpu().numpy()[i]][0].cpu().numpy()
                patches_top_5.append(patch_temp)
            patches_top_5_list.append(patches_top_5)
    coords_list = []
    for i in args.data_list:
        parent_image = get_parent_image([i], args)

        coords = get_patch_coordinate(patches_top_5, parent_image)
        coords_list.append(coords)
    output_dict = {}

    for i, j in enumerate(files):
        logging.info(
            f"File: {j}, PIRADS score: {pirads_list[i] + 2.0}, csPCa risk score: {cspca_risk_list[i]:.4f}"
        )
        output_dict[j] = {
            "Predicted PIRAD Score": pirads_list[i] + 2.0,
            "csPCa risk": cspca_risk_list[i],
            "Top left coordinate of top 5 patches(x,y,z)": coords_list[i],
        }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(output_dict, f, indent=4)
