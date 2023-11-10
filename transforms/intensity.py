from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any
from warnings import warn

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.data.meta_obj import get_track_meta
from monai.data.ultrasound_confidence_map import UltrasoundConfidenceMap
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.networks.layers import GaussianFilter, HilbertTransform, MedianFilter, SavitzkyGolayFilter
from monai.transforms.transform import RandomizableTransform, Transform
from monai.transforms.utils import Fourier, equalize_hist, is_positive, rescale_array
from monai.transforms.utils_pytorch_numpy_unification import clip, percentile, where
from monai.utils.enums import TransformBackends
from monai.utils.misc import ensure_tuple, ensure_tuple_rep, ensure_tuple_size, fall_back_tuple
from monai.utils.module import min_version, optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype

import monai
from monai.transforms.transform import Transform
from monai.utils import convert_to_tensor
from monai.data.meta_obj import get_track_meta
import numpy as np

class NanToZero:
    def __call__(self, tensor):
        if torch.isnan(tensor).any():
            tensor[torch.isnan(tensor)] = 0
        return tensor

class AddChannel(Transform):
    def __call__(self, img):
        out = convert_to_tensor(img[None], track_meta=get_track_meta())
        return out

class NormalizeIntensityUnbiased(monai.transforms.NormalizeIntensity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _std(x):
        if isinstance(x, np.ndarray):
            return np.std(x)
        x = torch.std(x.float(), unbiased=True)
        return x.item() if x.numel() == 1 else x

class RandHistogramShift(RandomizableTransform):
    """
    Apply random nonlinear transform to the image's intensity histogram.

    Args:
        num_control_points: number of control points governing the nonlinear intensity mapping.
            a smaller number of control points allows for larger intensity shifts. if two values provided, number of
            control points selecting from range (min_value, max_value).
        prob: probability of histogram shift.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, num_control_points = 10, prob: float = 0.1) -> None:
        RandomizableTransform.__init__(self, prob)

        if isinstance(num_control_points, int):
            if num_control_points <= 2:
                raise ValueError("num_control_points should be greater than or equal to 3")
            self.num_control_points = (num_control_points, num_control_points)
        else:
            if len(num_control_points) != 2:
                raise ValueError("num_control points should be a number or a pair of numbers")
            if min(num_control_points) <= 2:
                raise ValueError("num_control_points should be greater than or equal to 3")
            self.num_control_points = (min(num_control_points), max(num_control_points))
        self.reference_control_points: NdarrayOrTensor
        self.floating_control_points: NdarrayOrTensor

    def interp(self, x: NdarrayOrTensor, xp: NdarrayOrTensor, fp: NdarrayOrTensor) -> NdarrayOrTensor:
        ns = torch if isinstance(x, torch.Tensor) else np
        if isinstance(x, np.ndarray):
            # approx 2x faster than code below for ndarray
            return np.interp(x, xp, fp)

        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indices = ns.searchsorted(xp.reshape(-1), x.reshape(-1)) - 1
        indices = ns.clip(indices, 0, len(m) - 1)

        f = (m[indices] * x.reshape(-1) + b[indices]).reshape(x.shape)
        f[x < xp[0]] = fp[0]
        f[x > xp[-1]] = fp[-1]
        return f


    def randomize(self, data = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        num_control_point = self.R.randint(self.num_control_points[0], self.num_control_points[1] + 1)
        self.reference_control_points = np.linspace(0, 1, num_control_point)
        self.floating_control_points = np.copy(self.reference_control_points)
        for i in range(1, num_control_point - 1):
            self.floating_control_points[i] = self.R.uniform(
                self.floating_control_points[i - 1], self.floating_control_points[i + 1]
            )


    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        if self.reference_control_points is None or self.floating_control_points is None:
            raise RuntimeError("please call the `randomize()` function first.")
        img_t = convert_to_tensor(img, track_meta=False)
        img_min, img_max = img_t.min(), img_t.max()
        if img_max - img_min < 1e-7: # cannot use equality here due to numerical errors
            warn(
                f"The image's intensity is a single value {img_min}. "
                "The original image is simply returned, no histogram shift is done."
            )
            return img
        xp, *_ = convert_to_dst_type(self.reference_control_points, dst=img_t)
        yp, *_ = convert_to_dst_type(self.floating_control_points, dst=img_t)
        reference_control_points_scaled = xp * (img_max - img_min) + img_min
        floating_control_points_scaled = yp * (img_max - img_min) + img_min
        img_t = self.interp(img_t, reference_control_points_scaled, floating_control_points_scaled)
        return convert_to_dst_type(img_t, dst=img)[0]