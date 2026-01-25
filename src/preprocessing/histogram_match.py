import SimpleITK as sitk 
import os
import numpy as np 
import nrrd
from tqdm import tqdm
import pandas as pd
import random
import json
from skimage import exposure
import multiprocessing
import logging
def get_histmatched(data, ref_data, mask, ref_mask):

    source_pixels = data[mask > 0]
    ref_pixels = ref_data[ref_mask > 0]
    matched_pixels = exposure.match_histograms(source_pixels, ref_pixels)
    matched_img = data.copy()
    matched_img[mask > 0] = matched_pixels

    return matched_img

def histmatch(args):

    files = os.listdir(args.t2_dir)
    
    t2_histmatched_dir = os.path.join(args.output_dir, 't2_histmatched')
    dwi_histmatched_dir = os.path.join(args.output_dir, 'DWI_histmatched')
    adc_histmatched_dir = os.path.join(args.output_dir, 'ADC_histmatched')
    os.makedirs(t2_histmatched_dir, exist_ok=True)
    os.makedirs(dwi_histmatched_dir, exist_ok=True)
    os.makedirs(adc_histmatched_dir, exist_ok=True)
    logging.info("Starting histogram matching")
    for file in files:

        t2_image, header_t2 = nrrd.read(os.path.join(args.t2_dir, file))
        dwi_image, header_dwi = nrrd.read(os.path.join(args.dwi_dir, file))
        adc_image, header_adc = nrrd.read(os.path.join(args.adc_dir, file))

        ref_t2, _ = nrrd.read(os.path.join(args.project_dir, 'dataset', 't2_reference.nrrd'))
        ref_dwi, _ = nrrd.read(os.path.join(args.project_dir, 'dataset', 'dwi_reference.nrrd'))
        ref_adc , _ = nrrd.read(os.path.join(args.project_dir, 'dataset', 'adc_reference.nrrd'))

        prostate_mask, _ = nrrd.read(os.path.join(args.seg_dir, file))
        ref_prostate_mask, _ = nrrd.read(os.path.join(args.project_dir, 'dataset', 'prostate_segmentation_reference.nrrd'))

        histmatched_t2 = get_histmatched(t2_image, ref_t2, prostate_mask, ref_prostate_mask)
        histmatched_dwi = get_histmatched(dwi_image, ref_dwi, prostate_mask, ref_prostate_mask)
        histmatched_adc = get_histmatched(adc_image, ref_adc, prostate_mask, ref_prostate_mask)



        nrrd.write(os.path.join(t2_histmatched_dir, file), histmatched_t2, header_t2)
        nrrd.write(os.path.join(dwi_histmatched_dir, file), histmatched_dwi, header_dwi)
        nrrd.write(os.path.join(adc_histmatched_dir, file), histmatched_adc, header_adc)

        args.t2_dir = t2_histmatched_dir
        args.dwi_dir = dwi_histmatched_dir
        args.adc_dir = adc_histmatched_dir

    return args


