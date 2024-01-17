import torch  #深度学习框架，用于构建和训练神经网络。
import torch.nn as nn #神经网络相关的模块，提供了构建网络的基本构件如层、激活函数等。
import numpy as np #一种用于科学计算的库，主要用于处理高维数组和矩阵
import random
import json #用于读写JSON文件的库。
import matplotlib.pyplot as plt #一个绘图库，用于可视化训练过程中的准确率和损失


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size):
        super(TorchModel, self).__init__() #类的初始化函数。定义了模型的层，包括 nn.Embedding, nn.AvgPool1d, nn.Linear
        self.embedding = nn.Embedding(vocab_size, vector_dim)  # embedding层
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.classify = nn.Linear(vector_dim, sentence_length)  # 线性层，预测每个位置的字母
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        x = self.classify(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab(): #创建一个字母到索引的映射字典，用于之后的字符嵌入
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {char: index for index, char in enumerate(chars)}  # 为每个字生成一个标号
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):#生成单个样本，包括一个随机字符序列和其对应的二进制标签
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]  # 随机生成一个样本
    y = [1 if char in "abc" else 0 for char in x]  # 指定哪些字出现时为正样本
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


def build_dataset(sample_length, vocab, sentence_length): #生成一个数据集，该数据集由多个样本组成，每个样本由 build_sample 函数生成
    dataset_x = []  # 初始化一个空列表 dataset_x，用来存储数据集中的输入样本。
    dataset_y = []  # 初始化一个空列表 dataset_y，用来存储数据集中的目标标签。
    for _ in range(sample_length):  # 开始一个循环，循环 sample_length 次，以生成指定数量的样本。
        x, y = build_sample(vocab, sentence_length)  # 生成单个样本句子及其标签。
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(
        dataset_y)  # 在生成所有样本和标签后，将列表 dataset_x 和 dataset_y 转换成 PyTorch 张量
    # dataset_x 用于模型的输入，而 dataset_y 用于训练时的目标标签。


def build_model(vocab, char_dim, sentence_length): #根据提供的词汇表大小、字符维度和句子长度构建模型
    vocab_size = len(vocab)
    model = TorchModel(char_dim, sentence_length, vocab_size)
    return model


def evaluate(model, vocab, sample_length): #评估模型的性能，计算准确率
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    #  计算正样本的总数
    num_positive_samples = torch.sum(y == 1).item()
    #  计算负样本的总数
    num_negative_samples = torch.sum(y == 0).item()
    print("本次预测集中共有%d个正样本，%d个负样本" % (num_positive_samples, num_negative_samples))
    with torch.no_grad():
        y_pred = model(x)
        y_pred_class = torch.round(y_pred)
        accuracy = (y_pred_class == y).all(dim=1).float().mean().item()
    print("准确率：%f" % accuracy)
    return accuracy


#模型训练和评估的完整流程。
# 定义了训练周期、批次大小、样本数量、字符维度、句子长度和学习率。
# 使用Adam优化器进行优化，并在每个周期后评估模型的性能。
# 最后，绘制了准确率和损失的图形，并将模型和词汇表保存到文件中
def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    char_dim = 6  # 修改为六维
    sentence_length = 6
    learning_rate = 0.005
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()


if __name__ == "__main__":
    main()
