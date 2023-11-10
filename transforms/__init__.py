import monai
from .intensity import *
from .from_fmcib import *


def aug_transform(size, prob=0.5):
    if isinstance(size, int):
        size = [size] * 3
    elif not isinstance(size, tuple):
        raise TypeError(f"Unsupported type for size: {type(size)}")
        
    return monai.transforms.Compose([
        RandomResizedCrop3D(size=size[0]), # from_fmcib.py
        monai.transforms.RandAxisFlip(prob=prob),
        RandHistogramShift(prob=prob), # intensity.py
        monai.transforms.RandGaussianSmooth(prob=prob),
        monai.transforms.SpatialPad(spatial_size=size),
        NanToZero(),
    ])

def aug_transform_CT(size, prob=0.5):
    if isinstance(size, int):
        size = [size] * 3
    elif not isinstance(size, tuple):
        raise TypeError(f"Unsupported type for size: {type(size)}")

    return monai.transforms.Compose([
        monai.transforms.ToTensor(),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialPad(spatial_size=size),
        monai.transforms.RandGaussianSmooth(prob=prob),
        monai.transforms.RandAffine(prob=prob, translate_range=[10, 10, 10]),
        monai.transforms.RandAxisFlip(prob=prob),
        monai.transforms.RandRotate90(prob=prob),
        monai.transforms.NormalizeIntensity(subtrahend=-1024, divisor=3072),
        NanToZero(),
    ])