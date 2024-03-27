# -*- coding:utf-8 -*-
# @FileName : tmp.py
# @Time : 2024/3/27 13:32
# @Author : fiv

# 分词标注
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")