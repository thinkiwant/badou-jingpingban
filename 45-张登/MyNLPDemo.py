
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


"""
nlpDemo改造成6分类任务，a出现在第几位；换成RNN解决
"""


class TorchModel(nn.Module):
    def __init__(self, vocab, chars_len, char_dim):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), char_dim, padding_idx= 0 )
        #print(self.embedding.weight)
        self.RNN = nn.RNN(char_dim, chars_len+1, bias=False, batch_first=True)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        # x = x.transpose(1,2)
        # x = self.pool(x)
        x = self.RNN(x)[1]
        y_pred = x.squeeze()
        if y is not None:
            return self.loss(y_pred, y)       #loss的入参必须为float
        else:
            return y_pred

#字符集
def build_vocab():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 27
    return vocab

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_sample(vocab, length):
    chars = vocab
    result = []

    while len(result) < length:
        # 从字符集合中随机选取一个字符并添加到结果列表中
        char = random.choice(list(chars.keys()))

        if char not in result:
            result.append(char)
    if 'a' in result:
        y = result.index('a')+1
    else:
        y = 0

    x = [chars.get(word, chars['unk']) for word in result]  # 将字转换成序号，为了做embedding
    return x, y
#
#建立数据集 首先建立样本
def build_data(sum, vocab, length):
    data_x = []
    data_y = []
    for i in range(sum):
        x , y = build_sample(vocab, length)
        data_x.append(x)
        data_y.append(y)
    return torch.LongTensor(data_x), torch.LongTensor(data_y)

def build_Model(vocab, chars_len, char_dim):
    model = TorchModel(vocab, chars_len, char_dim)
    return model

def main():
    #配置参数
    train_num = 10        #训练轮数
    batch_size = 5       #每次训练样本个数
    train_sample = 1000   #每轮训练总共训练的样本总数
    char_dim = 10         #每个字的维度
    chars_len = 5   #样本文本长度
    learning_rate = 0.005 #学习率

    vocab = build_vocab()
    model = build_Model(vocab, chars_len ,char_dim)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for train in range(train_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_data(batch_size, vocab, chars_len)
            optim.zero_grad()
            loss = model(x, y)
            # loss = torch.tensor(loss, dtype=float,requires_grad=True)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========================\n第%d轮平均loss:%f" % (train+1, np.mean(watch_loss)))
        acc = test(model, vocab, chars_len)
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "modelweek3.pth")

    writer = open("vocab3.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def test(model, vocab, chars_len):
    model.eval()
    x, y = build_data(200, vocab, chars_len)
    # xy = y[(y > 0)]
    # print("本次预测集中共有%d个带a样本，%d个不带a样本" % (np.size(xy), 200 - np.size(xy)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def pre(vocab_path, model_path, input_strings):
    char_dim = 10  # 每个字的维度
    sentence_length = 5  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_Model(vocab, sentence_length, char_dim)  # 建立模型
    model.load_state_dict(torch.load(model_path))       # 加载训练好的权重

    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        no = np.argmax(result[i])
        print(result)
        print(result[i])
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, no, result[i][no])) # 打印结果

if __name__ == "__main__":
    main()
    test_strings = ["fnafe", "wzsfg", "aqdeg", "nkwwa"]
    pre("vocab3.json", "modelweek3.pth", test_strings)