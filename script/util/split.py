# -*- coding:utf-8 -*-
# @FileName : split.py
# @Time : 2024/3/21 19:46
# @Author : fiv

from pathlib import Path

from sklearn.model_selection import train_test_split


def split_dataset_dir(dataset_dir: Path):
    dataset_class = list(dataset_dir.glob("*"))
    # print(dataset_class)
    train_files, test_files = [], []
    for cla in dataset_class:
        if cla == "train" or cla == "test":
            continue
        files_path = list(cla.glob("*"))
        train, test = train_test_split(files_path, test_size=0.2)
        train_files.extend(train)
        test_files.extend(test)
    return train_files, test_files

# if __name__ == '__main__':
#     print(split_dataset(Path("../../data/dog")))
