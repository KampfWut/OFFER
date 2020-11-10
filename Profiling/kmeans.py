# -*- coding: utf-8 -*-
from sklearn import metrics
import random
import networkx as nx
import csv
import os
from sklearn.cluster import KMeans
import numpy as np


def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5) * (normB**0.5)) * 100, 2)


# node embedding得到向量表示
def read_embedding(file_path):
    vector_dic = {}
    with open(file_path, 'r') as f:
        for line_num, eachline in enumerate(f):
            if line_num == 1:
                continue
            for line in f:
                node = str(line.split(' ')[0])
                vector = []
                list_1 = line.split(' ')[1:]
                for i in list_1:
                    vector.append(float(i))
                vector_dic[node] = vector
    return vector_dic


# 创建样本
def create_sample(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as csvFile:
        for line in csvFile:
            node1 = str(line[:-1].split('\t')[0])
            node2 = str(line[:-1].split('\t')[1])
            G.add_edge(node1, node2)
    num = int(G.number_of_edges() / 2)                  # 取样数目
    positive_sample = random.sample(G.edges(), num)     # 从list中随机获取5个元素，作为一个片断返回
    neg_sample = random.sample(list(nx.non_edges(G)), num)
    for u, v in positive_sample:
        G.remove_edge(u, v)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    tagetlist = [positive_sample, neg_sample, num, G]
    return tagetlist


def eachFile(filepath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s\\%s' % (filepath, allDir))
        emblist = []
        for a, b in read_embedding(child).items():
            emblist.append(b)
        emb = np.array(emblist)
        km = KMeans(n_clusters=2, max_iter=100, random_state=0).fit(emb)
        labels = km.labels_
        A = metrics.silhouette_score(emb, labels,
                                     metric='euclidean')    # 轮廓系数 ，越大越好，[-1,+1]
        B = metrics.calinski_harabaz_score(emb,
                                           labels)          # Calinski-Harabasz分数，越大越好
        C = metrics.davies_bouldin_score(emb, labels)       # DBI index 越大越好
        print(child.decode('gbk'), A, B, C)


###
def onefile():
    child = "D:\\OnlineCode\\PythonCode\\OFFER\\Temp\\AttributeCharacteristics.emb"
    # AttributeCharacteristics, Combination[Bc], Combination[Bh], Combination[Bp]
    emblist = []
    for a, b in read_embedding(child).items():
        emblist.append(b)
    emb = np.array(emblist)
    # print(emb)
    km = KMeans(n_clusters=7, max_iter=500, random_state=0).fit(emb)
    labels = km.labels_
    A = metrics.silhouette_score(emb, labels,
                                 metric='euclidean')    # 轮廓系数 ，越大越好，[-1,+1]
    B = metrics.calinski_harabaz_score(emb, labels)     # Calinski-Harabasz分数，越大越好
    C = metrics.davies_bouldin_score(emb, labels)       # DBI index 越大越好
    print(">> Result: {}, {}, {}".format(A, B, C))


if __name__ == "__main__":
    onefile()
