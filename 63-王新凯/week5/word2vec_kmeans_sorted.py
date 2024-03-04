#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def get_mean_distance(centers, features, labels):
    dist = []
    nClass = len(centers)
    dist_sum = [0.0] * nClass
    feature_num = [0] * nClass
    for label, feature in zip(labels, features):
        dist_sum[label] += math.dist(centers[label], feature)
        feature_num[label] += 1
    for label in labels:
        dist_sum[label] /= feature_num[label]
    return dist_sum

def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    label_mean_dist = get_mean_distance(kmeans.cluster_centers_, vectors, kmeans.labels_)
    idxs = sorted(range(len(label_mean_dist)), key=lambda k:label_mean_dist[k]) #根据类间平均距升序排列
    sorted_dist = [label_mean_dist[i] for i in idxs]
    print("Rank clusters in the ascending order of mean intra-cluster distance:")
    order_i = 1
    for label, dist in zip(idxs, sorted_dist):
        sentences = sentence_label_dict[label]
        print(f"cluster {label} : mean dist : {dist} (order:{order_i})")
        order_i += 1
        for i in range(min(5, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")
if __name__ == "__main__":
    main()

