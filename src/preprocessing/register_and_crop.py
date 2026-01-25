import SimpleITK as sitk 
import os
import numpy as np 
import nrrd
from tqdm import tqdm
import pandas as pd
from picai_prep.preprocessing import PreprocessingSettings, Sample
import multiprocessing
from .center_crop import crop
import logging
def register_files(args):
    files = os.listdir(args.t2_dir)
    new_spacing = (0.4, 0.4, 3.0) 
    t2_registered_dir = os.path.join(args.output_dir, 't2_registered')
    dwi_registered_dir = os.path.join(args.output_dir, 'DWI_registered')
    adc_registered_dir = os.path.join(args.output_dir, 'ADC_registered')
    os.makedirs(t2_registered_dir, exist_ok=True)
    os.makedirs(dwi_registered_dir, exist_ok=True)
    os.makedirs(adc_registered_dir, exist_ok=True)
    logging.info("Starting registration and cropping")
    for file in tqdm(files):
            
        t2_image = sitk.ReadImage(os.path.join(args.t2_dir, file))
        dwi_image = sitk.ReadImage(os.path.join(args.dwi_dir, file))
        adc_image = sitk.ReadImage(os.path.join(args.adc_dir, file))

        original_spacing = t2_image.GetSpacing()
        original_size = t2_image.GetSize()
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]
    
        images_to_preprocess = {}
        images_to_preprocess['t2'] = t2_image
        images_to_preprocess['hbv'] = dwi_image
        images_to_preprocess['adc'] = adc_image

        pat_case = Sample(
            scans=[
                images_to_preprocess.get('t2'),
                images_to_preprocess.get('hbv'),
                images_to_preprocess.get('adc'),
            ],
            settings=PreprocessingSettings(spacing=[3.0,0.4,0.4], matrix_size=[new_size[2],new_size[1],new_size[0]]),
        )
        pat_case.preprocess()
    
        t2_post = pat_case.__dict__['scans'][0]
        dwi_post = pat_case.__dict__['scans'][1]
        adc_post = pat_case.__dict__['scans'][2]
        cropped_t2 = crop(t2_post, [args.margin, args.margin, 0.0])
        cropped_dwi = crop(dwi_post, [args.margin, args.margin, 0.0])
        cropped_adc = crop(adc_post, [args.margin, args.margin, 0.0])



        sitk.WriteImage(cropped_t2, os.path.join(t2_registered_dir, file))
        sitk.WriteImage(cropped_dwi, os.path.join(dwi_registered_dir, file))
        sitk.WriteImage(cropped_adc, os.path.join(adc_registered_dir, file))

        args.t2_dir = t2_registered_dir
        args.dwi_dir = dwi_registered_dir
        args.adc_dir = adc_registered_dir

        return args

