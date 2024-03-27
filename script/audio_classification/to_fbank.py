# -*- coding:utf-8 -*-
# @FileName : to_fbank.py
# @Time : 2024/3/20 14:14
# @Author : fiv

import torchaudio
from pathlib import Path

"""
Fbank：FilterBank：人耳对声音频谱的响应是非线性的，Fbank就是一种前端处理算法，
以类似于人耳的方式对音频进行处理，可以提高语音识别的性能。
获得语音信号的fbank特征的一般步骤是：预加重、分帧、加窗、短时傅里叶变换（STFT）、mel滤波、去均值等。
对fbank做离散余弦变换（DCT）即可获得mfcc特征。

MFCC(Mel-frequency cepstral coefficients):梅尔频率倒谱系数。
梅尔频率是基于人耳听觉特性提出来的， 它与Hz频率成非线性对应关系。
梅尔频率倒谱系数(MFCC)则是利用它们之间的这种关系，计算得到的Hz频谱特征。
主要用于语音数据特征提取和降低运算维度。例如：对于一帧有512维(采样点)数据，
经过MFCC后可以提取出最重要的40维(一般而言)数据同时也达到了降维的目的。
"""


def to_fbank(wav_path: Path):
    # from wav to fbank
    wav, sr = torchaudio.load(wav_path)
    fbank = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=40)
    # fbank = fbank.unsqueeze(0)
    return fbank

# from env import DATA_PATH
#
# print(to_fbank(DATA_PATH / "demo.wav").shape)
