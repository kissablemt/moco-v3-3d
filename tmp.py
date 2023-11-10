import transforms as my_transforms
import datasets as my_datasets
import monai
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl", init_method='tcp://localhost:10001',
                        world_size=1, rank=0)

def make_transform(patch_size):
    aug_block = my_transforms.aug_transform(size=patch_size)
    return monai.transforms.Compose([
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        my_transforms.Duplicate(transforms1=aug_block, transforms2=aug_block),
    ])

traindir = '/home/wzt/src/fmcibx/data/ROI_resize48/images_cut_48_resize'
train_dataset = my_datasets.SSLNoCropDataset(traindir, size=48, transform=make_transform(48), enable_negatives=False)
x, _ = train_dataset[0]
x1, x2 = x
print(x1.shape, x2.shape)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=(train_sampler is None),
    num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True)
print("train_loader: [OK]")

for i, (x, _) in enumerate(train_loader):
    x1, x2 = x
    print(i, x1.shape, x2.shape)
    break
