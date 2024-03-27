# -*- coding:utf-8 -*-
# @FileName : run.py
# @Time : 2024/3/21 16:48
# @Author : fiv

import torch
from dataset import get_dog_dataloader, get_animal_dataloader
from env import MODEL_PATH
from eval import eval
from model import Transformer
from train import train


def run_dog():
    torch.cuda.empty_cache()
    train_dataloader, test_dataloader = get_dog_dataloader()
    output_path = MODEL_PATH / "dog.pth"
    if output_path.exists():
        output_path.unlink()
    total_run = 1
    model = Transformer(n_out=3, num_encoder_layers=1, dropout=0.2)
    model = model.cuda()
    train(model, train_dataloader, total_run, output_path)
    model.load_state_dict(torch.load(output_path))
    eval(model, test_dataloader)


def run_animal():
    torch.cuda.empty_cache()
    train_dataloader, test_dataloader = get_animal_dataloader()
    output_path = MODEL_PATH / "animal.pth"
    if output_path.exists():
        output_path.unlink()
    total_run = 1
    model = Transformer(n_out=4, num_encoder_layers=1, dropout=0.2)
    model = model.cuda()
    train(model, train_dataloader, total_run, output_path)
    model.load_state_dict(torch.load(output_path))
    eval(model, test_dataloader)


if __name__ == '__main__':
    run_dog()
    run_animal()
