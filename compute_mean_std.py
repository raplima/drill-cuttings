"""computes mean and std of images in a folder
"""
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def compute_mean_std(folder, n_channels):
    """[summary]

    Args:
        folder ([type]): [description]
        n_channels ([type]): [description]
    """
    dataset = ImageFolder(folder,
                          transform=transforms.ToTensor())
    full_loader = DataLoader(dataset, shuffle=False,
                             num_workers=0)

    dset_mean = torch.zeros(n_channels)
    dset_std = torch.zeros(n_channels)

    print('==> Computing mean and std..')
    for inputs, _ in tqdm(full_loader):
        for i in range(n_channels):
            dset_mean[i] += inputs[:, i, :, :].mean()
            dset_std[i] += inputs[:, i, :, :].std()

    dset_mean.div_(len(dataset))
    dset_std.div_(len(dataset))

    print(f'\nmean: {dset_mean}')
    print(f'std: {dset_std}')
    return dset_mean, dset_std


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-fin', '--folder_in')
    parser.add_argument('-nchan', '--number_of_channels')

    args = parser.parse_args()

    compute_mean_std(args.folder_in,
                     args.number_of_channels,
                     )
