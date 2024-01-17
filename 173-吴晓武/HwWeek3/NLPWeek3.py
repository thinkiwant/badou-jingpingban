import torch
import torch.nn as nn
import numpy as np
import random
import json
import  matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 将 NLPDemo转为6分类任务，看a出现在第几个位置，用当前模型或者rnn来解决


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_lenght, vocab_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim) #将单词或者字符转为向量 嵌入层
        self.pool = nn.AvgPool1d(sentence_lenght)  #将向量进行池化 减少维度
        self.classify = nn.Linear(vector_dim, 6)   #  6分类

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.classify(x)
        return x

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0, "unk": 27} # 特殊字符 填充和未知
    for index, char in enumerate(chars):
        vocab[char] = index + 1 #为每一个字符配索引
    return vocab

def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if 'a' in x:
        y = x.index('a')  # 找到 'a' 的位置
    else:
        y = 5  # 如果 'a' 不在序列中，使用类别 5
    y = min(y, 5)  # 确保标签不超过5
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字符转换为索引
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab_size, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab_size)
    return model


def train_model(model, vocab, char_dim, sentence_length, epoch_num=10, batch_size=20, learning_rate=0.001):
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for _ in range(1000 // batch_size):  # 假设每个epoch有1000个样本
            x, y = build_dataset(batch_size, vocab, sentence_length)
            x_tensor = torch.LongTensor(x)
            y_tensor = torch.LongTensor(y)

            optim.zero_grad()
            y_pred = model(x_tensor)
            loss = loss_fn(y_pred, y_tensor)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / (1000 // batch_size)}")

    return model


def evaluate(model, vocab, sentence_length, test_size=200000):
    model.eval()  # 将模型设置为评估模式
    test_data_x, test_data_y = build_dataset(test_size, vocab, sentence_length)
    with torch.no_grad():  # 在这个块中，不计算梯度
        predictions = model(test_data_x)
        predicted_classes = torch.argmax(predictions, dim=1)
        correct_count = (predicted_classes == test_data_y).sum().item()
        accuracy = correct_count / test_size

    return accuracy

def main():
    # 设置参数
    char_dim = 20
    sentence_length = 6
    vocab = build_vocab()
    vocab_size = len(vocab)

    # 创建模型
    model = build_model(vocab_size, char_dim, sentence_length)

    # 训练模型
    model = train_model(model, vocab, char_dim, sentence_length)

    # 评估模型
    accuracy = evaluate(model, vocab, sentence_length)
    print(f"Test Accuracy: {accuracy}")

    # 预测新样本
    test_strings = ["abcdef", "ahijkl", "anopqr", "atuvwx", "azabcd"]
    predict(model, vocab, test_strings, sentence_length)

def predict(model, vocab, input_strings, sentence_length):
    model.eval()  # 将模型设置为评估模式
    for input_string in input_strings:
        # 处理输入字符串
        x = [vocab.get(char, vocab['unk']) for char in input_string]
        x = x[:sentence_length] + [vocab['pad']] * max(0, sentence_length - len(x))
        x_tensor = torch.LongTensor([x])

        # 进行预测
        with torch.no_grad():
            prediction = model(x_tensor)
            if prediction.dim() > 1:
                predicted_class = torch.argmax(prediction, dim=1).item()
            else:
                predicted_class = torch.argmax(prediction).item()



        print(f"输入: '{input_string}'，预测位置: {predicted_class}")


if __name__ == "__main__":
    main()
