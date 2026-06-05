import argparse
import logging
import os
import shutil

import nrrd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def smoothen_mask(args, sigma=1.0):
    files = os.listdir(args.seg_dir)

    for file in files:
        mask = sitk.ReadImage(os.path.join(args.seg_dir, file))
        mask = sitk.Cast(mask, sitk.sitkFloat32)

        # Smooth
        smoothed = sitk.DiscreteGaussian(mask, variance=1.5)

        # Threshold back to binary
        smooth_mask = smoothed > 0.5
        smooth_mask = sitk.Cast(smooth_mask, sitk.sitkUInt8)

        out_path = os.path.join(args.smooth_seg_dir_temp, file)
        sitk.WriteImage(smooth_mask, out_path)


def get_heatmap(args: argparse.Namespace) -> argparse.Namespace:
    """
    Generate heatmaps from DWI (Diffusion Weighted Imaging) and ADC (Apparent Diffusion Coefficient) medical imaging data.
    This function processes medical imaging files (DWI and ADC) along with their corresponding
    segmentation masks to create normalized heatmaps. It combines the DWI
    and ADC heatmaps through element-wise multiplication.
    Args:
        args: An object containing the following attributes:
            - t2_dir (str): Directory path containing T2 image files.
            - dwi_dir (str): Directory path containing DWI image files.
            - adc_dir (str): Directory path containing ADC image files.
            - seg_dir (str): Directory path containing segmentation mask files.
            - output_dir (str): Base output directory where 'heatmaps/' subdirectory will be created.
            - heatmapdir (str): Output directory for heatmap files (created by function).
    Returns:
        args: The modified args object with heatmapdir attribute set.
    Raises:
        FileNotFoundError: If input directories or files do not exist.
        ValueError: If NRRD files cannot be read properly.
    Notes:
        - DWI heatmap is normalized as (dwi - min) / (max - min)
        - ADC heatmap is normalized as (max - adc) / (max - min) (inverted)
        - Final heatmap is re-normalized to [0, 1] range
        - If all values in a mask region are identical, the heatmap is skipped for that modality
        - Output files are written in NRRD format with the same header as the input DWI file
    """

    files = os.listdir(args.t2_dir)
    args.heatmapdir = os.path.join(args.output_dir, "heatmaps/")
    os.makedirs(args.heatmapdir, exist_ok=True)
    args.smooth_seg_dir_temp = os.path.join(args.output_dir, "smooth_prostate_mask_temp/")
    os.makedirs(args.smooth_seg_dir_temp, exist_ok=True)
    args.smooth_seg_dir = os.path.join(args.output_dir, "smooth_prostate_mask/")
    os.makedirs(args.smooth_seg_dir, exist_ok=True)
    smoothen_mask(args)

    logging.info("Starting heatmap generation")
    for file in tqdm(files):
        bool_dwi = False
        bool_adc = False
        mask_temp, header_mask = nrrd.read(os.path.join(args.seg_dir, file))
        # spacing = np.linalg.norm(header_mask['space directions'], axis=1)
        dwi, header_dwi = nrrd.read(os.path.join(args.dwi_dir, file))
        adc, header_adc = nrrd.read(os.path.join(args.adc_dir, file))
        nonzero_vals_dwi = dwi[mask_temp > 0]
        mask, _ = nrrd.read(os.path.join(args.smooth_seg_dir_temp, file))
        mask = np.maximum(mask, mask_temp)

        if len(nonzero_vals_dwi) > 0:
            # min_val = nonzero_vals_dwi.min()
            # max_val = nonzero_vals_dwi.max()
            min_val, max_val = np.percentile(nonzero_vals_dwi, [1, 99])
            clipped_dwi = np.clip(dwi, min_val, max_val)

            heatmap_dwi = np.zeros_like(clipped_dwi, dtype=np.float32)

            if min_val != max_val:
                heatmap_dwi = (clipped_dwi - min_val) / (max_val - min_val)
                masked_heatmap_dwi = np.where(mask > 0, heatmap_dwi, 0)
            else:
                bool_dwi = True

        else:
            bool_dwi = True

        nonzero_vals_adc = adc[mask > 0]

        if len(nonzero_vals_adc) > 0:
            min_val = 0
            max_val = 1
            heatmap_adc = np.zeros_like(adc, dtype=np.float32)
            heatmap_adc = (max_val - adc) / (max_val - min_val)
            masked_heatmap_adc = np.where(mask > 0, heatmap_adc, 0)

        else:
            bool_adc = True

        if not bool_dwi and not bool_adc:
            mix_mask = (masked_heatmap_dwi * 0.3) + (masked_heatmap_adc * 0.7)
            # mix_mask = (masked_heatmap_dwi**0.5) * (masked_heatmap_adc**2.0)
            write_header = header_dwi
        elif bool_dwi:
            mix_mask = masked_heatmap_adc
            write_header = header_adc
        else:
            mix_mask = np.ones_like(adc, dtype=np.float32)
            write_header = header_dwi

        mix_mask = (mix_mask - mix_mask[mask > 0].min()) / (
            mix_mask[mask > 0].max() - mix_mask[mask > 0].min()
        )
        mix_mask = np.where(mask > 0, mix_mask, 0)
        nrrd.write(os.path.join(args.heatmapdir, file), mix_mask, write_header)
        nrrd.write(os.path.join(args.smooth_seg_dir, file), mask, header_mask)

    shutil.rmtree(args.smooth_seg_dir_temp)

    return args
