import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
from torchvision import transforms as T


def _load_dataset(base_path, dataset, include_controls=True):
    if isinstance(base_path, str):
        base_path = Path(base_path)

    df = pd.read_csv(base_path / (dataset + '.csv'))
    if include_controls:
        controls = pd.read_csv(base_path / (dataset + '_controls.csv'))
        df['well_type'] = 'treatment'
        df = pd.concat([controls, df], sort=True)
    df['cell_type'] = df.experiment.str.split("-").apply(lambda a: a[0])
    df['dataset'] = dataset
    dfs = []
    for site in (1, 2):
        df = df.copy()
        df['site'] = site
        dfs.append(df)
    res = pd.concat(dfs).sort_values(
        by=['id_code', 'site']).set_index('id_code')
    return res


def combine_metadata(base_path, include_controls=True):
    df = pd.concat(
        [
            _load_dataset(
                base_path, dataset, include_controls=include_controls)
            for dataset in ['test', 'train']
        ],
        sort=True)
    return df


class ImagesDataset(D.Dataset):
    def __init__(self, df, img_dir, mode='train', site=1, channels=[1, 2, 3, 4, 5, 6], transforms=None,
                 mixup=False, twohead=False, con_df=None):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.transforms = transforms
        self.mixup = mixup
        # export control image
        self.twohead = twohead
        if twohead:
            self.con_records = con_df[con_df['well_type']
                                      == 'negative_control'].to_records(index=False)

    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, index, channel, site):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])

    def _get_negacon(self, index, channel, site):
        # target sample
        experiment, plate = self.records[index].experiment, self.records[index].plate
        well = self.con_records[(self.con_records['experiment'] == experiment) *
                                (self.con_records['plate'] == plate)]['well'][0]
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])

    def __getitem__(self, index):

        if self.mixup:
            cindex = np.random.randint(0, self.len - 1)

        paths = []
        for site in self.site:
            paths.append([self._get_img_path(index, ch, site)
                          for ch in self.channels])
            if self.mixup:
                paths.append([self._get_img_path(cindex, ch, site)
                              for ch in self.channels])
            if self.twohead:
                paths.append([self._get_negacon(index, ch, site)
                              for ch in self.channels])

        imgs = []
        for path_i in paths:
            img_i = torch.cat([self._load_img_as_tensor(img_path)
                               for img_path in path_i])
            if self.transforms is not None:
                img_i = self.transforms(img_i)
            imgs.append(img_i)

        if self.mode == 'train':
            y = int(self.records[index].sirna)
            if self.mixup:
                mixed_imgs = []
                c = np.random.beta(a=0.2, b=0.2)
                cy = int(self.records[cindex].sirna)
                for site in range(len(self.site)):
                    mixed_imgs.append(
                        c * imgs[2 * site] + (1 - c) * imgs[2 * site + 1])
                # mixed_y = c * y + (1-c) * cy
                mixed_y = (y, cy, c)
                return (*mixed_imgs, mixed_y)
            else:
                return (*imgs, y)
        else:
            return (*imgs, self.records[index].id_code)

    def __len__(self):
        return self.len
