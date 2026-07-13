import os
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset

class SHGPairDataset(Dataset):
    def __init__(self, backward_dir, forward_dir):
        self.backward_dir = backward_dir
        self.forward_dir = forward_dir
        self.filenames = sorted([f for f in os.listdir(backward_dir) if f.endswith(".tif")])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        b_img = tiff.imread(os.path.join(self.backward_dir, fname)).astype(np.float32)
        f_img = tiff.imread(os.path.join(self.forward_dir, fname)).astype(np.float32)

        b_img = b_img[np.newaxis, :, :]
        f_img = f_img[np.newaxis, :, :]

        b_img = b_img * 2 - 1
        f_img = f_img * 2 - 1

        return torch.from_numpy(b_img), torch.from_numpy(f_img)