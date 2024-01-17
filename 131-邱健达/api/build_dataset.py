"""

用于存放所有人造数据集生成 函数 or 方法 的模块

"""
import random

import torch

# BuildNLPData_000.build_dataset(c:char, sentence_length:int, sample_length:int) -> X, Y
# y取值为第一次出现目标字符的下标 or 未出现时取sen_len
class BuildNLPData_000():
    @classmethod
    def build_vocab(cls):
        chars = "asdfghjklzxcvbnmwqertyuiop"
        vocab = {"padding": 0}
        for ix, elem in enumerate(chars):
            vocab[elem] = ix + 1
        vocab["unk"] = len(chars) + 1
        return vocab
    @classmethod
    def build_sample(cls, sentence_length, c):
        vocab = cls.build_vocab()
        # 这里不考虑padding, 让所有样本的长度一致为sentence_length
        x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
        y = sentence_length # 初始认为不在x的任何位置中
        for ix, elem in enumerate(x):
            if elem == c:
                y = ix
                break
        # 将x转化为序列
        x = [vocab.get(elem, vocab["unk"]) for elem in x]
        return x, y
    @classmethod
    def build_dataset(cls, c, sentence_length, sample_length):
        X = []
        Y = []
        for _ in range(sample_length):
            x, y = cls.build_sample(sentence_length, c)
            X.append(x)
            Y.append(y)
        return torch.LongTensor(X), torch.LongTensor(Y)
