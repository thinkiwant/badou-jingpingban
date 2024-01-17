#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from collections import Counter

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中某个特定字符出现的位置
规律：x是一个6维向量
    如果某个字符在第1位，属于第0类
    如果某个字符在第2位，属于第1类
    如果某个字符在第3位，属于第2类
    ....
    如果某个字符在第6位，属于第5类

"""

class TorchModel(nn.Module):
    # vector_dim 每个字的维度
    # sentence_length 样本文本长度
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.rnn = nn.RNN(sentence_length, vector_dim, bias=False, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length)     #线性层
        self.activation = torch.sigmoid     #sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  #loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        _, x  = self.rnn(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            y = y.float();
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

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

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，不能重复
    x = random.sample(list(vocab.keys()), sentence_length)  # 确保是唯一的 用sample
    # 查找a出现的位置索引
    if 'a' not in x:
        # 随机选择一个元素的索引
        index = random.randint(0, len(x) - 1)
        # 将随机选择的元素替换为 'a' 确保样本中‘a’真的存在
        x[index] = 'a'
    else:
        index = x.index('a')
    # 将字转换成序号，为了做embedding
    new_x = [vocab.get(word, vocab['unk']) for word in x]
    return new_x, index


#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    dataset_y = torch.nn.functional.one_hot(torch.tensor(dataset_y), sentence_length)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    # char_dim 每个字的维度
    # sentence_length 样本文本长度
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    # 将每一列转为元组，以便使用 Counter
    columns_as_tuples = [tuple(col.tolist()) for col in y]
    count_dict = Counter(columns_as_tuples)
    print('本次预测集中共有:')
    for key, value in count_dict.items():
        key = torch.tensor(key).argmax().item()
        print(f"{key} 类样本 {value} 个")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            index_of_max_t = np.argmax(y_t)
            index_of_max_p = np.argmax(y_p)
            if index_of_max_t == index_of_max_p:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 50        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            loss = model(x, y)   #计算loss
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
    torch.save(model.state_dict(), "model_my.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for vec, res in zip(input_strings, result):
        index_of_max = np.argmax(res)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, index_of_max, res[index_of_max]))  # 打印结果



if __name__ == "__main__":
    # main()
    test_strings = ["afnvfe", "wzcadg", "rqwadg", "nakwxw"]
    predict("model_my.pth", "vocab.json", test_strings)
