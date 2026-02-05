import argparse
import logging
import os

import nrrd
import numpy as np
import torch
from monai.bundle import ConfigParser
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
)
from monai.utils import set_determinism
from tqdm import tqdm

set_determinism(43)


def get_segmask(args: argparse.Namespace) -> argparse.Namespace:
    """
    Generate prostate segmentation masks using a pre-trained deep learning model.
    This function performs inference on T2-weighted MRI images to segment the prostate gland.
    It applies preprocessing transformations, runs the segmentation model, and saves the
    predicted masks. Post-processing is applied to retain only the top 10 slices with
    the highest non-zero voxel counts.
    Args:
        args: An arguments object containing:
            - output_dir (str): Base output directory where segmentation masks will be saved
            - project_dir (str): Root project directory containing model config and checkpoint
            - t2_dir (str): Directory containing input T2-weighted MRI images in NRRD format
    Returns:
        args: The updated arguments object with seg_dir added, pointing to the
              prostate_mask subdirectory within output_dir
    Raises:
        FileNotFoundError: If the model checkpoint or config file is not found
        RuntimeError: If CUDA operations fail on GPU
    Notes:
        - Automatically selects GPU (CUDA) if available, otherwise uses CPU
        - Applies MONAI transformations: loading, orientation (RAS), spacing (0.5mm isotropic),
          intensity scaling and normalization
        - Post-processing filters predictions to top 10 slices by non-zero voxel density
        - Output masks are saved in NRRD format preserving original image headers
    """

    args.seg_dir = os.path.join(args.output_dir, "prostate_mask")
    os.makedirs(args.seg_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    model = model_config.get_parsed_content("network_def").to(device)
    inferer = model_config.get_parsed_content("inferer")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

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
        _, header_t2 = nrrd.read(data["image"])
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

        nonzero_counts = np.count_nonzero(pred_nrrd, axis=(0, 1))
        top_slices = np.argsort(nonzero_counts)[-10:]
        output_ = np.zeros_like(pred_nrrd)
        output_[:, :, top_slices] = pred_nrrd[:, :, top_slices]

        nrrd.write(os.path.join(args.seg_dir, file), output_, header_t2)

    return args
