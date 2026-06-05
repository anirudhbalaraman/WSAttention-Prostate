import argparse
import logging
import os
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path

import yaml

from src.preprocessing.clip_intensity import clip_adc
from src.preprocessing.generate_heatmap import get_heatmap
from src.preprocessing.prostate_mask import get_segmask
from src.preprocessing.register_and_crop import register_files
from src.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="File preprocessing")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--steps",
        nargs="+",  # ← list of strings
        choices=[
            "register_and_crop",
            "histogram_match",
            "get_segmentation_mask",
            "get_heatmap",
        ],  # ← restrict allowed values
        help="Steps to execute (one or more)",
    )
    parser.add_argument("--t2_dir", default=None, help="Path to T2W files")
    parser.add_argument("--dwi_dir", default=None, help="Path to DWI files")
    parser.add_argument("--adc_dir", default=None, help="Path to ADC files")
    parser.add_argument("--seg_dir", default=None, help="Path to segmentation masks")
    parser.add_argument("--output_dir", default=None, help="Path to output folder")
    parser.add_argument(
        "--margin", default=0.2, type=float, help="Margin to center crop the images"
    )
    parser.add_argument("--project_dir", default=None, help="Project directory")

    args = parser.parse_args()

    if args.config:
        with open(args.config) as config_file:
            config = yaml.safe_load(config_file)
            args.__dict__.update(config)
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.project_dir is None:
        args.project_dir = Path(__file__).resolve().parent  # Set project directory

    FUNCTIONS: dict[str, Callable[[Namespace], Namespace]] = {
        "register_and_crop": register_files,
        "clip_adc": clip_adc,
        "get_segmentation_mask": get_segmask,
        "get_heatmap": get_heatmap,
    }

    args.logfile = os.path.join(args.output_dir, "preprocessing.log")
    setup_logging(args.logfile)
    logging.info("Starting preprocessing")
    if args.steps is None:
        args.steps = ["register_and_crop", "get_segmentation_mask", "clip_adc", "get_heatmap"]
    # validate_steps(args.steps)
    for step in args.steps:
        func = FUNCTIONS[step]
        args = func(args)
