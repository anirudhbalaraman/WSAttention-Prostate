import SimpleITK as sitk
import os
from tqdm import tqdm
from picai_prep.preprocessing import PreprocessingSettings, Sample
from .center_crop import crop
import logging


def register_files(args):
    """
    Register and crop medical images (T2, DWI, and ADC) to a standardized spacing and size.
    This function reads medical images from specified directories, resamples them to a
    new spacing of (0.4, 0.4, 3.0) mm, preprocesses them using the Sample class, and crops
    them with specified margins. The processed images are saved to new output directories.
    Args:
        args: An argument object containing:
            - t2_dir (str): Directory path containing T2 weighted images
            - dwi_dir (str): Directory path containing DWI (Diffusion Weighted Imaging) images
            - adc_dir (str): Directory path containing ADC (Apparent Diffusion Coefficient) images
            - output_dir (str): Directory path where registered images will be saved
            - margin (float): Margin in mm to crop from x and y dimensions
    Returns:
        args: Updated argument object with modified directory paths pointing to the
              registered image directories (t2_registered, DWI_registered, ADC_registered)
    Raises:
        FileNotFoundError: If input directories do not exist or files cannot be read
        RuntimeError: If image preprocessing or cropping fails
    """

    files = os.listdir(args.t2_dir)
    new_spacing = (0.4, 0.4, 3.0)
    t2_registered_dir = os.path.join(args.output_dir, "t2_registered")
    dwi_registered_dir = os.path.join(args.output_dir, "DWI_registered")
    adc_registered_dir = os.path.join(args.output_dir, "ADC_registered")
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
        images_to_preprocess["t2"] = t2_image
        images_to_preprocess["hbv"] = dwi_image
        images_to_preprocess["adc"] = adc_image

        pat_case = Sample(
            scans=[
                images_to_preprocess.get("t2"),
                images_to_preprocess.get("hbv"),
                images_to_preprocess.get("adc"),
            ],
            settings=PreprocessingSettings(
                spacing=[3.0, 0.4, 0.4], matrix_size=[new_size[2], new_size[1], new_size[0]]
            ),
        )
        pat_case.preprocess()

        t2_post = pat_case.__dict__["scans"][0]
        dwi_post = pat_case.__dict__["scans"][1]
        adc_post = pat_case.__dict__["scans"][2]
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
