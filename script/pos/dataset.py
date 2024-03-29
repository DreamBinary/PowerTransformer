# -*- coding:utf-8 -*-
# @FileName : dataset.py
# @Time : 2024/3/27 9:16
# @Author : fiv

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class POSDataset(Dataset):

    def __init__(self, vocabs, labels, max_length):
        self.states = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'Mg', 'm',
                       'Ng', 'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'o', 'p', 'q', 'Rg', 'r', 's', 'na', 'Tg', 't', 'u',
                       'Vg', 'v', 'vd', 'vn', 'vvn', 'w', 'Yg', 'y', 'z']
        self.label2idx = {state: idx + 1 for idx, state in enumerate(self.states)}
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext", use_fast=True)
        self.vocab = [
            self.tokenizer(vocab, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length,
                           is_split_into_words=True)["input_ids"][0] for vocab
            in vocabs]
        # add padding in label
        # self.labels = [torch.tensor([self.label2idx[label]  for label in a_labels] + ) for a_labels in labels]
        self.labels = [torch.tensor(
            [self.label2idx[label] for label in a_labels][:max_length] + [0] * (
                    max_length - len(a_labels)))
            for a_labels in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vocab[idx], self.labels[idx]

    def tag_size(self):
        return len(self.states)

    def vocab_size(self):
        return self.tokenizer.vocab_size


def get_data(corpus_path):
    vocabs, classes = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        for line in lines:
            vocab, label = [], []
            words = line.split(" ")
            for word in words:
                word = word.strip()
                if '/' not in word:
                    continue
                pos = word.index("/")
                if '[' in word and ']' in word:
                    vocab.append(word[1:pos])
                    label.append(word[pos + 1:-1])
                    break
                if '[' in word:
                    vocab.append(word[1:pos])
                    label.append(word[pos + 1:])
                    break
                if ']' in word:
                    vocab.append(word[:pos])
                    label.append(word[pos + 1:-1])
                    break
                vocab.append(word[:pos])
                label.append(word[pos + 1:])

            assert len(vocab) == len(label)
            vocabs.append(vocab)
            classes.append(label)
    return vocabs, classes


def get_dataloader(corpus_path, max_length=128, batch_size=32):
    from sklearn.model_selection import train_test_split
    vocabs, classes = get_data(corpus_path)

    # print(vocabs[0], classes[0])
    train_vocabs, test_vocabs, train_classes, test_classes = train_test_split(vocabs, classes, test_size=0.2)
    train_dataset = POSDataset(train_vocabs, train_classes, max_length)
    test_dataset = POSDataset(test_vocabs, test_classes, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    get_dataloader("../../data/corpus.txt")
    # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    # #
    # # # 输入的句子
    # sentence_list = ['迈向', '充满', '希望', '的', '新', '世纪', '——', '一九九八年', '新年', '讲话', '（', '附', '图片',
    #                  '１',
    #                  '张', '）']
    # sentence = "".join(sentence_list)
    # #
    # tokens = tokenizer(sentence_list, padding=True, truncation=True, return_tensors="pt", is_split_into_words=True,
    #                    add_special_tokens=False)
    # print(tokens)
    # print(len(tokens["input_ids"][0]))
    # print(len(sentence))
    # tokens = tokenizer(sentence, padding=False, truncation=True, return_tensors="pt",
    #                    add_special_tokens=False)
    #
    # print(tokens)
    # print(len(tokens["input_ids"][0]))
    # print(len(sentence))
    #
    # for sen in sentence_list:
    #     tokens = tokenizer(sen, padding=False, truncation=True, return_tensors="pt", add_special_tokens=False)
    #     print(tokens)
    #     print(len(tokens["input_ids"][0]), len(sen))

    # print(help(tokenizer.__call__))
    # n = 0
    # for x, y in dataloader:
    #     n += 1
    #     print(x.shape, y.shape)
    #     if n == 5:
    #         break
