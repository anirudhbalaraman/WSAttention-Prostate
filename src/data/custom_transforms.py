import numpy as np
import torch
from typing import Union
from monai.transforms import MapTransform
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.transform import Transform
from monai.transforms.utils import soft_clip
from monai.transforms.utils_pytorch_numpy_unification import clip, percentile
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor
from scipy.ndimage import binary_dilation
import cv2
from collections.abc import Hashable, Mapping, Sequence


class DilateAndSaveMaskd(MapTransform):
    """
    Custom transform to dilate binary mask and save a copy.
    """

    def __init__(self, keys, dilation_size=10, copy_key="original_mask"):
        super().__init__(keys)
        self.dilation_size = dilation_size
        self.copy_key = copy_key

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            mask = d[key].numpy() if isinstance(d[key], torch.Tensor) else d[key]
            mask = mask.squeeze(0)  # Remove channel dimension if present

            # Save a copy of the original mask
            d[self.copy_key] = torch.tensor(mask, dtype=torch.float32).unsqueeze(
                0
            )  # Save to a new key

            # Apply binary dilation to the mask
            dilated_mask = binary_dilation(mask, iterations=self.dilation_size).astype(np.uint8)

            # Store the dilated mask
            d[key] = torch.tensor(dilated_mask, dtype=torch.float32).unsqueeze(
                0
            )  # Add channel dimension back

        return d


class ClipMaskIntensityPercentiles(Transform):
    """
    Clip image intensity values based on percentiles computed from a masked region.
    This transform clips the intensity range of an image to values between lower and upper
    percentiles calculated only from voxels where the mask is positive. It supports both
    hard clipping and soft (smooth) clipping via a sharpness factor.
    Args:
        lower: Lower percentile threshold in range [0, 100]. If None, no lower clipping applied.
        upper: Upper percentile threshold in range [0, 100]. If None, no upper clipping applied.
        sharpness_factor: If provided, applies soft clipping with this sharpness parameter.
            Must be greater than 0. If None, applies hard clipping instead.
        channel_wise: If True, applies clipping independently to each channel using the
            corresponding channel's mask. If False, uses the same mask for all channels.
        dtype: Output data type for the clipped image. Defaults to np.float32.
    Raises:
        ValueError: If both lower and upper are None, if percentiles are outside [0, 100],
            if upper < lower, or if sharpness_factor <= 0.
    Returns:
        Clipped image with intensities adjusted based on masked percentiles.
    Note:
        Supports both torch.Tensor and numpy.ndarray inputs.


    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    """

    def __init__(
        self,
        lower: Union[float, None],
        upper: Union[float, None],
        sharpness_factor: Union[float, None] = None,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        if lower is None and upper is None:
            raise ValueError("lower or upper percentiles must be provided")
        if lower is not None and (lower < 0.0 or lower > 100.0):
            raise ValueError("Percentiles must be in the range [0, 100]")
        if upper is not None and (upper < 0.0 or upper > 100.0):
            raise ValueError("Percentiles must be in the range [0, 100]")
        if upper is not None and lower is not None and upper < lower:
            raise ValueError("upper must be greater than or equal to lower")
        if sharpness_factor is not None and sharpness_factor <= 0:
            raise ValueError("sharpness_factor must be greater than 0")

        # self.mask_data = mask_data
        self.lower = lower
        self.upper = upper
        self.sharpness_factor = sharpness_factor
        self.channel_wise = channel_wise
        self.dtype = dtype

    def _clip(self, img: NdarrayOrTensor, mask_data: NdarrayOrTensor) -> NdarrayOrTensor:
        masked_img = img * (mask_data > 0)
        if self.sharpness_factor is not None:
            lower_percentile = (
                percentile(masked_img, self.lower) if self.lower is not None else None
            )
            upper_percentile = (
                percentile(masked_img, self.upper) if self.upper is not None else None
            )
            img = soft_clip(
                img, self.sharpness_factor, lower_percentile, upper_percentile, self.dtype
            )
        else:
            lower_percentile = (
                percentile(masked_img, self.lower)
                if self.lower is not None
                else percentile(masked_img, 0)
            )
            upper_percentile = (
                percentile(masked_img, self.upper)
                if self.upper is not None
                else percentile(masked_img, 100)
            )
            img = clip(img, lower_percentile, upper_percentile)

        img = convert_to_tensor(img, track_meta=False)
        return img

    def __call__(self, img: NdarrayOrTensor, mask_data: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t = convert_to_tensor(img, track_meta=False)
        mask_t = convert_to_tensor(mask_data, track_meta=False)
        if self.channel_wise:
            img_t = torch.stack(
                [self._clip(img=d, mask_data=mask_t[e]) for e, d in enumerate(img_t)]
            )  # type: ignore
        else:
            img_t = self._clip(img=img_t, mask_data=mask_t)

        img = convert_to_dst_type(img_t, dst=img)[0]

        return img


class ClipMaskIntensityPercentilesd(MapTransform):
    """
    Dictionary wrapper for ClipMaskIntensityPercentiles.
    Args:
        keys: Keys of the corresponding items to be transformed.
        mask_key: Key to the mask data in the input dictionary used to compute percentiles. Only intensity values where the mask is positive will be considered.
        lower: Lower percentile value (0-100) for clipping. If None, no lower clipping is applied.
        upper: Upper percentile value (0-100) for clipping. If None, no upper clipping is applied.
        sharpness_factor: Optional factor to enhance contrast after clipping. If None, no sharpness enhancement is applied.
        channel_wise: If True, compute percentiles separately for each channel. If False, compute globally.
        dtype: Data type of the output. Defaults to np.float32.
        allow_missing_keys: If True, missing keys will not raise an error. Defaults to False.
    Example:
        >>> transform = ClipMaskIntensityPercentilesd(
        ...     keys=["image"],
        ...     mask_key="mask",
        ...     lower=2,
        ...     upper=98,
        ...     sharpness_factor=1.0
        ... )
    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str,
        lower: Union[float, None],
        upper: Union[float, None],
        sharpness_factor: Union[float, None] = None,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ClipMaskIntensityPercentiles(
            lower=lower,
            upper=upper,
            sharpness_factor=sharpness_factor,
            channel_wise=channel_wise,
            dtype=dtype,
        )
        self.mask_key = mask_key

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key], d[self.mask_key])
        return d


class ElementwiseProductd(MapTransform):
    """
    A dictionary-based transform that computes the elementwise product of two arrays.
    This transform multiplies two input arrays element-by-element and stores the result
    in a specified output key.
    Args:
        keys: Collection of keys to select from the input dictionary. Must contain exactly
            two keys whose corresponding values will be multiplied together.
        output_key: Key in the output dictionary where the product result will be stored.
    Returns:
        Dictionary with the elementwise product stored at the output_key.
    Example:
        >>> transform = ElementwiseProductd(keys=["image1", "image2"], output_key="product")
        >>> data = {"image1": np.array([1, 2, 3]), "image2": np.array([2, 3, 4])}
        >>> result = transform(data)
        >>> result["product"]
        array([ 2,  6, 12])
    """

    def __init__(self, keys: KeysCollection, output_key: str) -> None:
        super().__init__(keys)
        self.output_key = output_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        d[self.output_key] = d[self.keys[0]] * d[self.keys[1]]
        return d


class CLAHEd(MapTransform):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to images in a data dictionary.
    Works on 2D images or 3D volumes (applied slice-by-slice).

    Args:
        keys (KeysCollection): Keys of the items to be transformed.
        clip_limit (float): Threshold for contrast limiting. Default is 2.0.
        tile_grid_size (Union[tuple, Sequence[int]]): Size of grid for histogram equalization (default: (8,8)).
    """

    def __init__(
        self,
        keys: KeysCollection,
        clip_limit: float = 2.0,
        tile_grid_size: Union[tuple, Sequence[int]] = (8, 8),
    ) -> None:
        super().__init__(keys)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image_ = d[key]

            image = image_.cpu().numpy()

            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            # Handle 2D images or process 3D images slice-by-slice.

            image_clahe = np.stack([clahe.apply(slice) for slice in image[0]])

            # Convert back to float in [0,1]
            processed_img = image_clahe.astype(np.float32) / 255.0
            reshaped_ = processed_img.reshape(1, *processed_img.shape)
            d[key] = torch.from_numpy(reshaped_).to(image_.device)
        return d


class NormalizeIntensity_custom(Transform):
    """
    Normalize input based on the `subtrahend` and `divisor`: `(img - subtrahend) / divisor`.
    Use calculated mean or std value of the input image if no `subtrahend` or `divisor` provided.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    When `channel_wise` is True, the first dimension of `subtrahend` and `divisor` should
    be the number of image channels if they are not None.
    If the input is not of floating point type, it will be converted to float32

    Args:
        subtrahend: the amount to subtract by (usually the mean).
        divisor: the amount to divide by (usually the standard deviation).
        nonzero: whether only normalize non-zero values.
        channel_wise: if True, calculate on each channel separately, otherwise, calculate on
            the entire image directly. default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        subtrahend: Union[Sequence, NdarrayOrTensor, None] = None,
        divisor: Union[Sequence, NdarrayOrTensor, None] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype

    @staticmethod
    def _mean(x):
        if isinstance(x, np.ndarray):
            return np.mean(x)
        x = torch.mean(x.float())
        return x.item() if x.numel() == 1 else x

    @staticmethod
    def _std(x):
        if isinstance(x, np.ndarray):
            return np.std(x)
        x = torch.std(x.float(), unbiased=False)
        return x.item() if x.numel() == 1 else x

    def _normalize(
        self, img: NdarrayOrTensor, mask_data: NdarrayOrTensor, sub=None, div=None
    ) -> NdarrayOrTensor:
        img, *_ = convert_data_type(img, dtype=torch.float32)
        """
        if self.nonzero:
            slices = img != 0
            masked_img = img[slices]
            if not slices.any():
                return img
        else:
            slices = None
            masked_img = img
        """
        slices = None
        mask_data = mask_data.squeeze(0)
        slices_mask = mask_data > 0
        masked_img = img[slices_mask]

        _sub = sub if sub is not None else self._mean(masked_img)
        if isinstance(_sub, (torch.Tensor, np.ndarray)):
            _sub, *_ = convert_to_dst_type(_sub, img)
            if slices is not None:
                _sub = _sub[slices]

        _div = div if div is not None else self._std(masked_img)
        if np.isscalar(_div):
            if _div == 0.0:
                _div = 1.0
        elif isinstance(_div, (torch.Tensor, np.ndarray)):
            _div, *_ = convert_to_dst_type(_div, img)
            if slices is not None:
                _div = _div[slices]
            _div[_div == 0.0] = 1.0

        if slices is not None:
            img[slices] = (masked_img - _sub) / _div
        else:
            img = (img - _sub) / _div
        return img

    def __call__(self, img: NdarrayOrTensor, mask_data: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        mask_data = convert_to_tensor(mask_data, track_meta=get_track_meta())
        dtype = self.dtype or img.dtype
        if self.channel_wise:
            if self.subtrahend is not None and len(self.subtrahend) != len(img):
                raise ValueError(
                    f"img has {len(img)} channels, but subtrahend has {len(self.subtrahend)} components."
                )
            if self.divisor is not None and len(self.divisor) != len(img):
                raise ValueError(
                    f"img has {len(img)} channels, but divisor has {len(self.divisor)} components."
                )

            if not img.dtype.is_floating_point:
                img, *_ = convert_data_type(img, dtype=torch.float32)

            for i, d in enumerate(img):
                img[i] = self._normalize(  # type: ignore
                    d,
                    mask_data,
                    sub=self.subtrahend[i] if self.subtrahend is not None else None,
                    div=self.divisor[i] if self.divisor is not None else None,
                )
        else:
            img = self._normalize(img, mask_data, self.subtrahend, self.divisor)

        out = convert_to_dst_type(img, img, dtype=dtype)[0]
        return out


class NormalizeIntensity_customd(MapTransform):
    """
    Dictionary-based wrapper of :class:`NormalizeIntensity_custom`.

    The mean and standard deviation are calculated only from intensities which are
    defined in the mask provided through ``mask_key``.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.MapTransform`
        mask_key: key of the corresponding mask item to be used for calculating
            statistics (mean and std).
        subtrahend: the amount to subtract by (usually the mean). If None,
            the mean is calculated from the masked region of the input image.
        divisor: the amount to divide by (usually the standard deviation). If None,
            the std is calculated from the masked region of the input image.
        nonzero: whether only normalize non-zero values.
        channel_wise: if True, calculate on each channel separately, otherwise, calculate on
            the entire image directly. Defaults to False.
        dtype: output data type, if None, same as input image. Defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = NormalizeIntensity_custom.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str,
        subtrahend: Union[NdarrayOrTensor, None] = None,
        divisor: Union[NdarrayOrTensor, None] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = NormalizeIntensity_custom(
            subtrahend, divisor, nonzero, channel_wise, dtype
        )
        self.mask_key = mask_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalizer(d[key], d[self.mask_key])
        return d
