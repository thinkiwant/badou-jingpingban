"""

基于embedding + rnn/pooling + linear的简单nlp任务实现
任务1: 预测指定字符在正向遍历字符串的首次出现位置 < len(字符串)个任务2模型 or input - embedding - rnn - linear - softmax >
任务2: 预测指定字符在字符串中的指定位置是否出现 < len(字符串)个任务3模型 >
任务3: 预测指定字符在字符串中是否出现 < input - embedding - pooling - linear - sigmoid >
...

"""
import torch
import torch.nn as nn
import numpy as np
import api.build_dataset as build
import api.plot_DL as plt
import api.data_tools as fileOp

class RnnModel(nn.Module):
    def __init__(self, sentence_length, hidden_size, embedding_dim, vocab):
        super().__init__()
        self.embedding = nn.Embedding(embedding_dim=embedding_dim,
                                      num_embeddings=len(vocab))
        self.linear = nn.Linear(hidden_size, sentence_length+1) # h_t: hidden_size * 1 -> y_p: sen_len+1
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          bias=False,
                          batch_first=True)
        self.activation = torch.softmax
        self.loss = nn.functional.cross_entropy # 第一个batch变量给概率分布, 第二个batch变量给LongTensor类别
    def forward(self, x, y_r=None):
        x = self.embedding(x) # exp shape: batch_size(20) * sen_len(4) * embedding_dim(28)
        H, h_end = self.rnn(x) # h_t: batch_size(20) * hidden_size(5)
                               # H: batch_size(20) * sen_len(4) * hidden_size(5)
        y_p = self.linear(h_end.squeeze()) # batch_size(20) * (sen_len(4)+1)(5)
        if y_r is None:
            y_p = torch.argmax(self.activation(y_p, 0))
            return y_p
        else:
            loss = self.loss(y_p, y_r)
            return loss

def evaluate(model, sen_len):
    model.eval()
    x_test, y_test = build.BuildNLPData_000.build_dataset('a',
                                                          sentence_length=sen_len,
                                                          sample_length=200)
    correct = wrong = 0
    with torch.no_grad():
        for x, y_r in zip(x_test, y_test):
            y_p = model(x)
            if int(y_p) == int(y_r):
                correct += 1
            else:
                wrong += 1
    return correct / (correct + wrong)
def predict(model_path, vocab, vec):
    model = RnnModel(sentence_length=len(vec[0]),
                     hidden_size=len(vec[0])+1,
                     embedding_dim=len(vocab),
                     vocab=vocab)
    model = fileOp.pthLoad(model, model_path)
    for ix, elem in enumerate(vec):
        x = [vocab.get(k, vocab["unk"]) for k in elem]
        print(f"sentence: {elem}\n"
              f"y_p: {model(torch.LongTensor(x))}")

def train(vocab, sen_len):
    # 超参数
    sample_total = 5000
    epoch_size = 100
    batch_size = 20
    lr = .0005
    embedding_dim = len(vocab) # 词表长度 种 可能的取值维度, 保证不同取值维度lir < 不要有a=1, b=2, a+a=b的含义 >
    hidden_size = sen_len + 1
    # 训练数据建立 - 1000个长度为sen_len的样本, y_t取值为sen_len时为不存在目标字母
    x_t, y_t = build.BuildNLPData_000.build_dataset("a",
                                                    sentence_length=sen_len,
                                                    sample_length=sample_total)
    # 选择模型和优化器
    model = RnnModel(sentence_length=sen_len,
                     embedding_dim=embedding_dim,
                     hidden_size=hidden_size,
                     vocab=vocab)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 训练过程 和 损失记录
    accAndce = []
    for epoch in range(epoch_size):
        loss_batch = []
        for batch_ix in range(sample_total // batch_size):
            x = x_t[batch_ix * batch_size: (batch_ix + 1) * batch_size]
            y = y_t[batch_ix * batch_size: (batch_ix + 1) * batch_size]
            model.train()
            loss_cur = model(x, y)
            loss_cur.backward()
            optim.step()
            optim.zero_grad()
            loss_batch.append(loss_cur.item())
        ce_epoch = np.mean(loss_batch)
        print("="*5 + f"{epoch}轮所有batch的ce均值:{ce_epoch}")
        acc_test = evaluate(model, sen_len)
        print("="*5 + f"{epoch}轮测试集准确率:{acc_test}")
        accAndce.append([acc_test, ce_epoch])
    # 保存模型
    fileOp.pthSave(model, "model3.pth")
    return accAndce
if __name__ == "__main__":
    # 建词表
    # vocab = build.BuildNLPData_000.build_vocab()
    # fileOp.dictToJson(vocab, "vocab3.json")
    # 读词表
    vocab = fileOp.readFromJson("vocab3.json")
    # 训练并绘制训练结果
    sen_len = 50
    plt.plotACC_testAndLOSS_train(train(vocab, sen_len))
    # 预测集
    vec10 = ["9da0c39dp0",
             "-x1a0c39dp",
             "]ocp0c39dp"]
    # predict("model3.pth", vocab, vec10)
    vec50 = ["pfenpa9jc2.vmc='.csp"
             "pfenpy9jc2.vmc='.csp"
             "pfenpg9jc2"]
    predict("model3.pth", vocab, vec50)