#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是某个特定字符出现的位置

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab,hidden_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)  # 添加 RNN 层
        self.classify = nn.Linear(hidden_size, sentence_length)     #线性层
        self.loss = nn.CrossEntropyLoss  #loss函数采用交叉熵

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)
        x,_ = self.rnn(x)
        x = self.classify(x[:,-1,:])
        return x

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 在0-6中随机替换'a'
    random_index = torch.randint(0, 6, (1,))
    x[random_index] = 'a'
    # 寻找目标字符的位置，如果不存在则返回-1
    y = x.index('a')
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, y


#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length,hidden_size):
    model = TorchModel(char_dim, sentence_length, vocab,hidden_size)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    total_correct = 0
    with torch.no_grad():
        output = model(x,y)      #模型预测
        predicted = torch.argmax(output,dim=1) #将y_pred用argmax转化成索引
        total_correct += (predicted == y).sum().item() #predicted==y的值统计并转化为标量
    accuracy = total_correct / len(x)
    print(f"正确预测个数：{total_correct}, 正确率：{accuracy*100}%")
    return accuracy


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    hidden_size=6
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length,hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #  损失函数
    loss_fn = nn.CrossEntropyLoss()
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            output = model(x,y)
            loss = loss_fn(output, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    hidden_size = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length,hidden_size)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        output = model.forward(torch.LongTensor(x))  #模型预测
        target_index = torch.argmax(output, dim=1)
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string,  round(target_index[i].item()), target_index[i])) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["anvfee", "wasdfg", "rqkdea", "nadwww",'jhusaa','okjasj']
    predict("model.pth", "vocab.json", test_strings)
