import os
from typing import Union
import SimpleITK as sitk
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from tqdm import tqdm
from AIAH_utility.viewer import BasicViewer, ListViewer
from PIL import Image
import monai
from monai.bundle import ConfigParser
from monai.config import print_config
import torch
import sys
import os
import nibabel as nib
import shutil

from tqdm import trange, tqdm

from monai.data import DataLoader, Dataset, TestTimeAugmentation, create_test_image_2d
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    CropForegroundd,
    DivisiblePadd,
    Invertd,
    LoadImaged,
    ScaleIntensityd,
    RandRotated,
    RandRotate,
    InvertibleTransform,
    RandFlipd,
    Activations,
    AsDiscrete,
    NormalizeIntensityd,
)
from monai.utils import set_determinism
from monai.transforms import (
    Resize,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    EnsureTyped,
)
import nrrd

set_determinism(43)
from monai.data import MetaTensor
import SimpleITK as sitk
import pandas as pd
import logging
def get_segmask(args):

    args.seg_dir = os.path.join(args.output_dir, "prostate_mask")
    os.makedirs(args.seg_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config_file = os.path.join(args.project_dir, "config", "inference.json")
    model_config = ConfigParser()
    model_config.read_config(model_config_file)
    model_config["output_dir"] = args.seg_dir
    model_config["dataset_dir"] = args.t2_dir
    files = os.listdir(args.t2_dir)
    model_config["datalist"] = [os.path.join(args.t2_dir, f) for f in files]
    

    checkpoint = os.path.join(
        args.project_dir,
        "models",
        "prostate_segmentation_model.pt",
    )
    preprocessing = model_config.get_parsed_content("preprocessing")
    model = model_config.get_parsed_content("network_def").to(device)
    inferer = model_config.get_parsed_content("inferer")
    postprocessing = model_config.get_parsed_content("postprocessing")
    dataloader = model_config.get_parsed_content("dataloader")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    keys = "image"
    transform = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=[0.5, 0.5, 0.5], mode="bilinear"),
            ScaleIntensityd(keys=keys, minv=0, maxv=1),
            NormalizeIntensityd(keys=keys),
            EnsureTyped(keys=keys),
        ]
    )
    logging.info("Starting prostate segmentation")
    for file in tqdm(files):

        data = {"image": os.path.join(args.t2_dir, file)}
        transformed_data = transform(data)
        a = transformed_data
        with torch.no_grad():
            images = a["image"].reshape(1, *(a["image"].shape)).to(device)
            data["pred"] = inferer(images, network=model)
        pred_img = data["pred"].argmax(1).cpu()

        model_output = {}
        model_output["image"] = MetaTensor(pred_img, meta=transformed_data["image"].meta)
        transformed_data["image"].data = model_output["image"].data
        temp = transform.inverse(transformed_data)
        pred_temp = temp["image"][0].numpy()
        pred_nrrd = np.round(pred_temp)
        
        nonzero_counts = np.count_nonzero(pred_nrrd, axis=(0,1))
        top_slices = np.argsort(nonzero_counts)[-10:]
        output_ = np.zeros_like(pred_nrrd)
        output_[:,:,top_slices] = pred_nrrd[:,:,top_slices]

        nrrd.write(os.path.join(args.seg_dir, file), output_)

        return args


    