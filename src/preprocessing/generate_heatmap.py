
import os
import numpy as np
import nrrd
import json
import pandas as pd
import json
import SimpleITK as sitk
import multiprocessing

import logging


def get_heatmap(args):

    files = os.listdir(args.t2_dir)
    args.heatmapdir = os.path.join(args.output_dir, 'heatmaps/')
    os.makedirs(args.heatmapdir, exist_ok=True)
    for file in files:

        bool_dwi = False
        bool_adc = False
        mask, _ = nrrd.read(os.path.join(args.seg_dir, file))
        dwi, _ = nrrd.read(os.path.join(args.dwi_dir, file))  
        adc, _ = nrrd.read(os.path.join(args.adc_dir, file))

        nonzero_vals_dwi = dwi[mask > 0]
        if len(nonzero_vals_dwi) > 0:
            min_val = nonzero_vals_dwi.min()
            max_val = nonzero_vals_dwi.max()
            heatmap_dwi = np.zeros_like(dwi, dtype=np.float32)
            
            if min_val != max_val:
                heatmap_dwi = (dwi - min_val) / (max_val - min_val)
                masked_heatmap_dwi = np.where(mask > 0, heatmap_dwi, heatmap_dwi[mask>0].min())
            else:
                bool_dwi = True
                
        else:
            bool_dwi = True

        nonzero_vals_adc = adc[mask > 0]
        if len(nonzero_vals_adc) > 0:
            min_val = nonzero_vals_adc.min()
            max_val = nonzero_vals_adc.max()
            heatmap_adc = np.zeros_like(adc, dtype=np.float32)
            
            if min_val != max_val:
                heatmap_adc = (max_val - adc) / (max_val - min_val)
                masked_heatmap_adc = np.where(mask > 0, heatmap_adc, heatmap_adc[mask>0].min())
            else:
                bool_adc = True

        else:
            bool_adc = True
            

        if bool_dwi:
            mix_mask = masked_heatmap_adc  
        if bool_adc:
            mix_mask = masked_heatmap_dwi 
        if not bool_dwi and not bool_adc:
            mix_mask = masked_heatmap_dwi * masked_heatmap_adc
        else:
            mix_mask = np.ones_like(adc, dtype=np.float32)

        mix_mask = (mix_mask - mix_mask.min()) / (mix_mask.max() - mix_mask.min())


        nrrd.write(os.path.join(args.heatmapdir, file), mix_mask)
    
    return args


    
    