# -*- coding:utf-8 -*-
# @FileName : dataset.py
# @Time : 2024/3/27 9:16
# @Author : fiv
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class POSDataset(Dataset):

    def __init__(self, vocabs, labels, max_length):
        self.states = ['NONE', 'Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
                       'Mg', 'm', 'Ng', 'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'o', 'p', 'q', 'Rg', 'r', 's', 'na', 'Tg',
                       't', 'u', 'Vg', 'v', 'vd', 'vn', 'vvn', 'w', 'Yg', 'y', 'z']
        self.label2idx = {state: idx for idx, state in enumerate(self.states)}
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext", use_fast=True)
        self.corpus = [self.token_and_align_labels(vocab, label, max_length) for vocab, label in zip(vocabs, labels)]

    def token_and_align_labels(self, tokens, labels, max_length):
        tokens = self.tokenizer(tokens, truncation=True, is_split_into_words=True, padding="max_length",
                                add_special_tokens=False, max_length=max_length)
        word_ids = tokens.word_ids()
        aligned_labels = []
        for wid in word_ids:
            if wid is None:
                aligned_labels.append("NONE")
            else:
                aligned_labels.append(labels[wid])
        # return tokens["input_ids"], aligned_labels
        return torch.tensor(tokens["input_ids"]), torch.tensor([self.label2idx[label] for label in aligned_labels])

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus[idx]

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
    # get_dataloader("../../data/corpus.txt")
    corpus_path = "../../data/corpus_demo.txt"
    train_dataloader, test_dataloader = get_dataloader(corpus_path, batch_size=1)
    for x, y in train_dataloader:
        print(len(x), len(y))
    # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    # # #
    # # 迈向/v 充满/v 希望/n 的/u 新/a 世纪/n ——/w 一九九八年/t 新年/t 讲话/n （/w 附/v 图片/n １/m 张/q ）/w
    # sentence_list = ['迈向', '充满', '希望', '的', '新', '世纪', '——', '一九九八年', '新年', '讲话', '（', '附', '图片',
    #                  '１', '张', '）']
    # lable_list = ['v', 'v', 'n', 'u', 'a', 'n', 'w', 't', 't', 'n', 'w', 'v', 'n', 'm', 'q', 'w']
    # #
    # tokens = tokenizer(sentence_list, truncation=True, is_split_into_words=True, padding="max_length",
    #                    add_special_tokens=False, max_length=128)
    # word_ids = tokens.word_ids()
    # print(word_ids)
    # aligned_labels = []
    # for id in word_ids:
    #     if id is None:
    #         aligned_labels.append(-100)
    #     else:
    #         aligned_labels.append(lable_list[id])
    # print(tokens)
    # print(aligned_labels)
    # print(word_ids)
    #
    # print(len(word_ids))
    # print(len(lable_list))
    #
    # print(len(tokens["input_ids"]))
    # print(len(sentence_list))
    # print(len(aligned_labels))

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
