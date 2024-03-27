# -*- coding:utf-8 -*-
# @FileName : eval.py
# @Time : 2024/3/21 16:31
# @Author : fiv
import torch


def eval(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            # print(y, predicted)
    acc = correct / total
    print(f"Accuracy: {acc:.4f}")
