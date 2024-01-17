import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符a所在位置进行分类
对比rnn和pooling做法

"""
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True, num_layers=1)
        self.classify = nn.Linear(vector_dim, 7)  # 改造为6分类任务
        self.activation = nn.LogSoftmax(dim=1)  # 使用LogSoftmax作为输出层激活函数

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded)
        last_hidden = outputs[:, -1, :]  # 取最后一个
        y_pred = self.classify(last_hidden)
        return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本，目标标签是"a"在句子中的位置（0-5）
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    target_index = x.index('a') if 'a' in x else 6  # 若'a'不在序列中，则默认为位置6（可以自定义处理缺失情况）
    y = torch.tensor(target_index, dtype=torch.long)
    x = [vocab.get(word, vocab['pad']) for word in x]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab_size, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab_size)
    return model


# 测试代码
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个样本"%(len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    vocab = build_vocab()

    model = build_model(len(vocab), char_dim, sentence_length)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.view(-1, 7), y.squeeze(1))  # reshape输出适应CrossEntropyLoss
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab,  sentence_length)
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(len(vocab), char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) #打印结果


if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wzsdfg", "rqwdeg", "nakwww"]
    predict("model.pth", "vocab.json", test_strings)