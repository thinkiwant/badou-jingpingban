import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中字母c出现的位置

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, vocab):
        super(TorchModel, self).__init__()
        # embedding 层
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.linear = nn.Linear(vector_dim, len(vocab))
        self.pool = nn.AvgPool1d(vector_dim)
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        embedded_output = self.embedding(x)
        pool_output = self.pool(embedded_output)
        pool_output = pool_output.squeeze()
        linear_output = self.linear(pool_output)
        if self.training:  # 训练模式，返回损失值
            y = y.squeeze()
            return self.loss(linear_output, y)
        else:
            return torch.softmax(linear_output, dim=1)

# 构建字符表
def build_vocab():
    chars = "abcdefghijk"  #字符集
    vocab = {"pad":0}
    # 每个字对应一个序号
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab


# 在字符串中找到目标字符第一次出现时的对应位置，如果不存在对应字符，返回nuk 对应的下标
def find_position(matrix, target, max_index):
    for index, char in enumerate(matrix):
        if target in char:
            return index
    return max_index


# 创建样本数据
def build_sample(vocab, sentence_length):
    if sentence_length <= 0:
        raise ValueError("Sentence length should be greater than 0.")
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 当且仅当遇到第一个 c 时，返回 c 对应的下标
    y = [find_position(x, "c", vocab['unk'])]
    # 将字转换成序号，为了做embedding
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


def build_dataset(batch_size, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(batch_size):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim):
    model = TorchModel(char_dim, vocab)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    total_sample_num = 100
    x_sample, y_sample = build_dataset(total_sample_num, vocab, sample_length)
    flatten_y_sample = np.array(y_sample).flatten()
    with torch.no_grad():
        y_pred = model(x_sample)
        # 使用 torch.max 在每一行中找到最大值及其索引，即最可能情况对应的索引
        output_pred = np.array(torch.max(y_pred, dim=1)[-1])
        correct_totals = np.sum(output_pred == flatten_y_sample)
        wrong_totals = total_sample_num - correct_totals
        print(f"预测正确总计：{correct_totals}，预测错误总计：{wrong_totals}, 正确率：{correct_totals / total_sample_num}")
    return correct_totals / total_sample_num

def main():
    #配置参数
    epoch_num = 500        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    sentence_length = 10   #样本文本长度
    learning_rate = 0.01 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(sentence_length, vocab)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) # 构造一组训练样本
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return log


def draw(log):
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()


def predict(model_path, vocab_path, input_strings):
    sentence_length = 10
    # 加载字符表
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    # 模型初始化
    model = TorchModel(sentence_length, vocab)
    model.load_state_dict(torch.load(model_path))
    # 测试模式
    model.eval()
    x = []
    for input_string in input_strings:
        x.append([vocab[char] if char in vocab else vocab["unk"] for char in input_string])
    x_sample = torch.LongTensor(x)
    # 不计算梯度
    with torch.no_grad():
        y_pred = model(x_sample)
        output_pred = np.array(torch.max(y_pred, dim=1)[-1])
        for x, output in zip(x_sample, output_pred):
            print(f"输入的数据：{x}，预测结果：{output}")


if __name__ == "__main__":
    log = main()
    predict("model.pth", "vocab.json", ["abeeeecthc", "cabggggbbt"])
    draw(log)