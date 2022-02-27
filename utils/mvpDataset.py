import torch
import numpy as np
import h5py
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class ShapeNetH5(Dataset):
    def __init__(self, train=True, npoints=2048, npose = 26, novel_input=True, novel_input_only=False):
        if train:
            self.input_path = 'data/mvp/mvp_train_input.h5'
            self.gt_path = 'data/mvp/mvp_train_gt_%dpts.h5' % npoints
        else:
            self.input_path = 'data/mvp/mvp_test_input.h5'
            self.gt_path = 'data/mvp/mvp_test_gt_%dpts.h5' % npoints
        self.npoints = npoints
        self.train = train
        self.npose = npose

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        self.novel_input_data = np.array(input_file['novel_incomplete_pcds'][()])
        self.novel_labels = np.array(input_file['novel_labels'][()])
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array(gt_file['complete_pcds'][()])
        self.novel_gt_data = np.array(gt_file['novel_complete_pcds'][()])
        gt_file.close()

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        if self.npose != 26:
            pose_idx = np.linspace(0, 25, self.npose, dtype=np.int)
            sample_idx = np.arange(self.gt_data.shape[0]).reshape(-1, 1)*26
            sample_idx = (sample_idx + pose_idx).reshape(-1)

            self.input_data = self.input_data[sample_idx]
            self.labels = self.labels[sample_idx]

        print('Dataset: ',self.input_data.shape, self.gt_data.shape, self.labels.shape)

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // self.npose]))
        label = self.labels[index]
        return partial, label, complete

class MvpDataModule(pl.LightningDataModule):
    def __init__(self, npoints=2048, batch_size = 32, npose = 26, novel_input=False):
        super().__init__()
        self.npoints = npoints
        self.batch_size = batch_size
        self.npose = npose
        self.novel_input = novel_input

    def setup(self, stage=None):
        if stage =='fit' or stage is None:
            self.train_dataset = ShapeNetH5(train=True, npoints=self.npoints, npose=self.npose, novel_input=self.novel_input, novel_input_only=False)

            self.val_dataset = ShapeNetH5(train=False, npoints=self.npoints, npose=1, novel_input=self.novel_input, novel_input_only=False)

        if stage == 'test' or stage is None:
            self.test_dataset =  ShapeNetH5(train=False, npoints=self.npoints, npose=1, novel_input=self.novel_input, novel_input_only=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)
