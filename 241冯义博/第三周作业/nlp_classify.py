import random

import torch
import torch.nn as nn
import numpy as np

data = ["a", "b", "c", "d", "e", "f"]

vocab = {
    "a" : 0,
    "b" : 1,
    "c" : 2,
    "d" : 3,
    "e" : 4,
    "f" : 5,
}


def conv(s):
    return [vocab[i] for i in s]


def build_sample(size):
    x = []
    y = []
    for i in range(size):
        random.shuffle(data)
        s = "".join(data)
        a = [0] * len(data)
        index = s.index("a")
        a[index] = 1
        x.append(conv(s))
        y.append(a)
    return torch.IntTensor(x), torch.FloatTensor(y)



class IndexNet(nn.Module):

    def __init__(self):
        super(IndexNet, self).__init__()
        self.embedding = nn.Embedding(6, 6)
        self.layer = nn.RNN(6, 6, bias=False, batch_first=True)
        self.active = torch.sigmoid
        self.loss = nn.functional.mse_loss


    def forward(self, x, y=None):
        x = self.embedding(x)
        output, h = self.layer(x)
        y_pre = self.active(h)
        y_pre = y_pre.squeeze()
        if y is None:
            return y_pre
        else:
            return self.loss(y_pre, y)



def evaluate(net):
    net.eval()
    x, y = build_sample(100)
    pre = net(x)
    success = 0
    wrong = 0
    for p, r in zip(pre, y):
        p = nn.functional.softmax(p)
        p = p.detach().numpy()
        if np.argmax(p) == np.argmax(r):
            success += 1
        else:
            wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (success, success / (success + wrong)))





def train():
    epoch_num = 40
    batch_size = 20
    sample_size = 10000
    lr = 1e-3
    net = IndexNet()
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    for i in range(epoch_num):
        net.train()
        x, y = build_sample(sample_size)
        watch_loss = []
        for j in range(sample_size // batch_size):
            train_x = x[j * batch_size: (j + 1) * batch_size]
            train_y = y[j * batch_size: (j + 1) * batch_size]
            loss = net(train_x, train_y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(watch_loss)))
        evaluate(net)


    torch.save(net.state_dict(), "model.pt")





def predict(model_path, s):
    net = IndexNet()
    net.load_state_dict(torch.load(model_path))
    s = torch.IntTensor(conv(s))
    net.eval()
    with torch.no_grad():
        pre = net(s)
    index = np.argmax(pre)
    y = [0] * 6
    y[index] = 1
    print(y)





if __name__ == "__main__":
    train()
    s = "bcadef"
    predict("model.pt", s)
