import argparse
import logging
import os

import nrrd
import numpy as np
from tqdm import tqdm


def clip_adc(args: argparse.Namespace, adc_min=0.0, adc_max=3500.0):
    files = os.listdir(args.adc_dir)
    clip_adc_dir = os.path.join(args.output_dir, "ADC_clipped")
    os.makedirs(clip_adc_dir, exist_ok=True)
    logging.info("Starting clipping ADC")

    for file in tqdm(files):
        adc, header_adc = nrrd.read(os.path.join(args.adc_dir, file))

        if np.percentile(adc, 99) < 100:
            adc = adc * 100

        adc_clipped = np.clip(adc, adc_min, adc_max)
        adc_normalized = adc_clipped / adc_max
        nrrd.write(os.path.join(clip_adc_dir, file), adc_normalized, header_adc)

    args.adc_dir = clip_adc_dir
    return args
