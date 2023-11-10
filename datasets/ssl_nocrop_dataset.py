from pathlib import Path
from typing import Union, Tuple
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
import monai

try:
    from transforms import NormalizeIntensityUnbiased
except:
    import sys
    sys.path.append("/home/wzt/src/fmcibx")
    from transforms import NormalizeIntensityUnbiased


class SSLNoCropDataset(torch.utils.data.Dataset):
    def __init__(self, positive_dir, negative_dir=None, size: Union[int, Tuple[int]] = 48, mask_dir=None, enable_negatives=True, transform=None):
        super().__init__()

        self.positive_paths = list(Path(positive_dir).glob("*.nii.gz"))
        self.negative_paths = list(Path(negative_dir).glob("*.nii.gz")) if negative_dir else None
        self.mask_paths = list(Path(mask_dir).glob("*.nii.gz")) if mask_dir else None

        if isinstance(size, int):
            self.size = (size, size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise TypeError(f"Unsupported type for size: {type(size)}")

        self._len = len(self.positive_paths)
        self.enable_negatives = enable_negatives
        self.transform = transform

    def __len__(self):
        return self._len

    def read_image(self, path, normalize=False):
        img = sitk.ReadImage(path)
        img_np = sitk.GetArrayFromImage(img)
        if normalize:
            # img_np = NormalizeIntensityUnbiased()(img_np)
            img_np = monai.transforms.NormalizeIntensity()(img_np)
        return img_np

    def read_mask(self, path):
        mask_np = sitk.GetArrayFromImage(sitk.ReadImage(path))
        mask_np = mask_np.astype("int32")
        mask_np[mask_np > 0] = 1
        return mask_np

    def get_negative_patch(self):
        img_np = self.read_image(np.random.choice(self.negative_paths))
        return monai.transforms.ToTensor()(img_np)

    def get_positive_patch(self, index):
        img_np = self.read_image(self.positive_paths[index])
        return monai.transforms.ToTensor()(img_np)

    def __getitem__(self, index):
        # Get Label
        target = False

        # Get Positive Patch
        pos_patch = self.get_positive_patch(index)
        pos_patch = self.transform(pos_patch) if self.transform else pos_patch

        # Get Negative Patch
        if self.enable_negatives:
            neg_patch = self.get_negative_patch()
            neg_patch = self.transform(neg_patch) if self.transform else neg_patch
            return {"positive": pos_patch, "negative": neg_patch}, target # SSL Dataset does not need label
        return pos_patch, target

    def get_row(self, index):
        return dict(image_path=self.positive_paths[index], 
                    negative_path=self.negative_paths[index], 
                    mask_path=self.mask_paths[index])

    def get_mask(self, index):
        mask_np = self.read_mask(self.mask_paths[index])
        return monai.transforms.ToTensor()(mask_np)


def test():
    positive_dir = "/media/wzt/plum14t/wzt/ProcressedData/ROI_resize48/images_cut_48_resize"
    negative_dir = "/media/wzt/plum14t/wzt/ProcressedData/ROI_resize48/negative/"
    mask_dir = "/media/wzt/plum14t/wzt/ProcressedData/ROI_resize48/labels_cut_48_resize"
    enable_negatives = False
    size = 48
    idx = 1000

    dataset = SSLNoCropDataset(positive_dir, negative_dir, mask_dir=mask_dir, size=size, enable_negatives=enable_negatives)
    print("len: ", len(dataset))
    
    row = dataset.get_row(idx)
    print("image_path: ", row.get("image_path", "None"))
    print("mask_path: ", row.get("mask_path", "None"))
    
    x = dataset[idx]
    if enable_negatives:
        print(x["positive"].shape, x["negative"].shape)
        # sitk.WriteImage(sitk.GetImageFromArray(x["positive"]), f"/mnt/tmp/1/patch_{idx}_pos.nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(x["negative"]), f"/mnt/tmp/1/patch_{idx}_neg.nii.gz")
    else:
        print(x.shape)
        # sitk.WriteImage(sitk.GetImageFromArray(x), f"/mnt/tmp/1/patch_{idx}_pos.nii.gz")

    mask_patch = dataset.get_mask(idx)
    # sitk.WriteImage(sitk.GetImageFromArray(mask_patch), f"/mnt/tmp/1/patch_{idx}_mask.nii.gz")


if __name__ == '__main__':
    test()
    ...