import torch
from torch.utils.data import Dataset
import h5py

class H5ImageTextDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path

    def __len__(self):
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            return len(h5_file['imageA'])

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            group_names = list(h5_file.keys())
            sample_name = list(h5_file[group_names[0]].keys())[idx]
            imageA = torch.from_numpy(h5_file['imageA'][sample_name][()])
            imageB = torch.from_numpy(h5_file['imageB'][sample_name][()])
            text = torch.from_numpy(h5_file['text'][sample_name][()])
            return imageA, imageB, text, sample_name

