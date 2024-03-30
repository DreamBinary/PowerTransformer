# -*- coding:utf-8 -*-
# @FileName : train.py
# @Time : 2024/3/20 16:13
# @Author : fiv
import torch
from torch import nn
from tqdm import tqdm

from env import PATH


#
def train(model, pos_tag_size, dataloader, total_run=10, output_path=PATH.MODEL_PATH / "no_nome.pth"):
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    pbar = tqdm(range(total_run))
    min_loss = 100
    for _ in pbar:
        tot_loss = 0
        for x, y in dataloader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            # print(output.shape)
            # print(y.shape)
            # print(output)
            # print(y)
            output = output.view(-1, pos_tag_size)
            y = y.view(-1)
            # print(output.shape)
            # print(y.shape)
            # print(output)
            # print(y)
            loss = criterion(output, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot_loss += loss.item()
        tot_loss = tot_loss / len(dataloader)
        pbar.set_description(f"loss: {tot_loss:.4f}")
        if tot_loss < min_loss:
            min_loss = tot_loss
            torch.save(model.state_dict(), output_path)
    print(f"Min loss: {min_loss}")
