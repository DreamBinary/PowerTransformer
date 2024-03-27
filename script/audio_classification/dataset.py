# -*- coding:utf-8 -*-
# @FileName : dataset.py
# @Time : 2024/3/20 16:31
# @Author : fiv
import torch
from torch.utils.data import Dataset, DataLoader
from to_fbank import to_fbank


class AnimalDataset(Dataset):
    def __init__(self, dataset_dir=None):
        self.label2idx = {"bird": 0, "cat": 1, "dog": 2, "tiger": 3}
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        if dataset_dir is None:
            from env import DATA_PATH
            self.dataset_dir = DATA_PATH / "animal" / "all"
        else:
            self.dataset_dir = dataset_dir
        self.file_path = list(self.dataset_dir.glob("*.wav"))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        fbank = to_fbank(self.file_path[idx])
        length = 512
        if fbank.shape[0] < length:
            # repeat fbank to fill length
            fbank = torch.cat([fbank] * (length // fbank.shape[0] + 1), dim=0)[:length]
        else:
            start = torch.randint(0, fbank.shape[0] - length, (1,))
            fbank = fbank[start:start + length]
        label = self.file_path[idx].stem.split("_")[0]
        return fbank, self.label2idx[label]

    def idx2label(self, idx):
        return self.idx2label[idx]


def get_animal_dataloader(dataset_dir=None, batch_size=8, shuffle=True):
    if dataset_dir is None:
        from env import DATA_PATH
        dataset_dir = DATA_PATH / "animal"

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    train = AnimalDataset(train_dir)
    test = AnimalDataset(test_dir)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


class DogDataset(Dataset):
    def __init__(self, file_path):
        self.label2idx = {"adult": 0, "dogs": 1, "puppy": 2}
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.file_path = file_path

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        fbank = to_fbank(self.file_path[idx])
        length = 512
        if fbank.shape[0] < length:
            # repeat fbank to fill length
            fbank = torch.cat([fbank] * (length // fbank.shape[0] + 1), dim=0)[:length]
        else:
            start = torch.randint(0, fbank.shape[0] - length, (1,))
            fbank = fbank[start:start + length]
        label = self.file_path[idx].stem.split("_")[0]
        return fbank, self.label2idx[label]

    def idx2label(self, idx):
        return self.idx2label[idx]


def get_dog_dataloader(batch_size=8, shuffle=True):
    from script.util import split_dataset_dir
    from env import DATA_PATH
    dataset_dir = DATA_PATH / "dog"
    train_files, test_files = split_dataset_dir(dataset_dir)
    train_dataset = DogDataset(train_files)
    test_dataset = DogDataset(test_files)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader
