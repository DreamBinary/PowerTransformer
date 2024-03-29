# -*- coding:utf-8 -*-
# @FileName : run.py
# @Time : 2024/3/27 16:09
# @Author : fiv
import torch

from dataset import get_dataloader
from env import PATH
from eval import eval
from model import Transformer
from train import train


def run(corpus_path):
    torch.cuda.empty_cache()
    max_len = 32
    train_dataloader, test_dataloader = get_dataloader(corpus_path, batch_size=1, max_length=max_len)
    vocab_size = train_dataloader.dataset.vocab_size()
    tag_size = train_dataloader.dataset.tag_size()

    output_path = PATH.MODEL_PATH.value / "pos.pth"
    if output_path.exists():
        output_path.unlink()

    total_run = 1
    model = Transformer(vocab_size=vocab_size, pos_tag_size=tag_size, max_length=max_len)
    model = model.cuda()
    train(model, train_dataloader, total_run, output_path)
    model.load_state_dict(torch.load(output_path))
    eval(model, test_dataloader)


if __name__ == '__main__':
    corpus_path = "../../data/corpus_demo.txt"
    run(corpus_path)

# from model import Transformer
# from dataset import get_dataloader
#
# dataloader = get_dataloader("../../data/corpus.txt", batch_size=1)
# vocab_size = dataloader.dataset.vocab_size()
# tag_size = dataloader.dataset.tag_size()
# model = Transformer(vocab_size=vocab_size, pos_tag_size=tag_size)
# model.train()
# n = 0
# for x, y in dataloader:
#     n += 1
#     print(x)
#     output = model(x)
#     print(output)
#     print(output.shape)
#     print(y)
#     print(y.shape)
#     if n == 5:
#         break
