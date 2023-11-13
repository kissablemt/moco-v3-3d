from typing import Union, Tuple
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
import monai
import scipy

try:
    from transforms import NormalizeIntensityUnbiased
except:
    import sys
    sys.path.append("/home/wzt/src/fmcibx")
    from transforms import NormalizeIntensityUnbiased


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, size: Union[int, Tuple[int]] = 48, label: str = None, enable_negatives=True, transform=None):
        super().__init__()

        self.csv_path = csv_path
        self.label = label
        self.enable_negatives = enable_negatives
        self.transform = transform

        if isinstance(size, int):
            self.size = (size, size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise TypeError(f"Unsupported type for size: {type(size)}")

        self._df = pd.read_csv(self.csv_path)
        self._len = len(self._df)
        self.enable_negatives = enable_negatives
        self.transform = transform

    def __len__(self):
        return self._len

    def read_image(self, path, normalize=True):
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

    def get_negative_patch(self, img_np):
        valid_patch_size = monai.data.utils.get_valid_patch_size(img_np.shape, self.size)
        neg_patch = img_np[monai.data.utils.get_random_patch(img_np.shape, valid_patch_size)]
        return monai.transforms.ToTensor()(neg_patch)

    def get_positive_patch(self, img_np, roi_center, roi_size):
        crop_size = max(max(roi_size), self.size[0])
        cropper = monai.transforms.Compose([
            monai.transforms.ToTensor(),
            monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
            monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=crop_size),
            monai.transforms.Resize(spatial_size=self.size) if crop_size > self.size[0] else monai.transforms.Lambda(lambda x: x),
            monai.transforms.SqueezeDim(dim=0),
        ])
        return cropper(img_np) 

    def __getitem__(self, index):
        row = self._df.iloc[index]
        
        # Read Image
        img = self.read_image(row.image_path, normalize=False)

        # Get 3D Bounding Box, in ITK-SNAP, axis-x is top-left, axis-y is bottom-right, axis-z is top-right
        roi_start = [row.roi_start_x, row.roi_start_y, row.roi_start_z]
        roi_end = [row.roi_end_x, row.roi_end_y, row.roi_end_z]
        roi_center = [(roi_start[i] + roi_end[i]) // 2 for i in range(3)]
        roi_size = [roi_end[i] - roi_start[i] for i in range(3)]
        
        # Get Label
        target = int(row[self.label]) if self.label is not None else False

        # Get Positive Patch
        pos_patch = self.get_positive_patch(img, roi_center, roi_size)
        pos_patch = self.transform(pos_patch) if self.transform else pos_patch

        # Get Negative Patch
        if self.enable_negatives:
            neg_patch = self.get_negative_patch(img)
            neg_patch = self.transform(neg_patch) if self.transform else neg_patch
            return {"positive": pos_patch, "negative": neg_patch}, target
        return pos_patch, target

    def get_row(self, index):
        return self._df.iloc[index]

    def get_mask(self, index):
        row = self._df.iloc[index]
        
        mask = self.read_mask(row.mask_path)

        # Get 3D Bounding Box, in ITK-SNAP, axis-x is top-left, axis-y is bottom-right, axis-z is top-right
        roi_start = [row.roi_start_x, row.roi_start_y, row.roi_start_z]
        roi_end = [row.roi_end_x, row.roi_end_y, row.roi_end_z]
        roi_center = [(roi_start[i] + roi_end[i]) // 2 for i in range(3)]
        roi_size = [roi_end[i] - roi_start[i] for i in range(3)]

        # Get Positive Patch
        mask_patch = self.get_positive_patch(mask, roi_center, roi_size)
        return mask_patch


def test():
    csv_path = "/home/wzt/src/fmcibx/data/imm_pre_unenhanced/labels_bbox_test.csv"
    enable_negatives = True
    size = 50
    label = "label"
    idx = 10

    dataset = SSLDataset(csv_path=csv_path, label="label", size=size, enable_negatives=enable_negatives)
    print("len: ", len(dataset))
    
    row = dataset.get_row(idx)
    print("image_path: ", row.get("image_path", "None"))
    print("mask_path: ", row.get("mask_path", "None"))
    
    x, label = dataset[idx]
    if enable_negatives:
        print(x["positive"].shape, x["negative"].shape, label)
        sitk.WriteImage(sitk.GetImageFromArray(x["positive"]), f"/mnt/tmp/1/patch_{idx}_pos.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(x["negative"]), f"/mnt/tmp/1/patch_{idx}_neg.nii.gz")
    else:
        print(x.shape, label)
        sitk.WriteImage(sitk.GetImageFromArray(x), f"/mnt/tmp/1/patch_{idx}_pos.nii.gz")

    mask_patch = dataset.get_mask(idx)
    sitk.WriteImage(sitk.GetImageFromArray(mask_patch), f"/mnt/tmp/1/patch_{idx}_mask.nii.gz")


if __name__ == '__main__':
    test()
    ...