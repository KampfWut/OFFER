# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
import networkx as nx
import random

SAMPLE_Count = 3000

#计算余弦相似度作为预测标准
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)

#node embedding得到向量表示
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
    
#创建样本
def create_sample(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as csvFile:
        for line in csvFile:
            node1 = str(line[:-1].split('\t')[0])
            node2 = str(line[:-1].split('\t')[1])
            G.add_edge(node1, node2)
    num = int(G.number_of_edges()/2) #取样数目
    print num
    positive_sample = random.sample(G.edges(), num)  #从list中随机获取元素，作为一个片断返回
    neg_sample = random.sample(list(nx.non_edges(G)), num)
    return positive_sample,neg_sample,num

if __name__ == '__main__':
    graph_path = '.\Dataset\soc-hamsterster.edges'
    emb_path = 'result.embeddings'
    num = create_sample(graph_path)[2]
    positive_sample = create_sample(graph_path)[0]
    neg_sample = create_sample(graph_path)[1]
    pos_list = [1]*num
    pos_list.extend([0]*num)
    y_true = np.array(pos_list)
    y_scores=[]
    vvdict= read_embedding('result.embeddings')
    for i in positive_sample:
        y_scores.append(cosine_similarity(vvdict.get(i[0]),vvdict.get(i[1])))
    for i in neg_sample:
        y_scores.append(cosine_similarity(vvdict.get(i[0]),vvdict.get(i[1])))
    y_score = np.array(y_scores)
    print "AUC is",metrics.roc_auc_score(y_true, y_score)
    threshold = np.average(y_score)
    y_score2 = [int(item>threshold) for  item in y_score]
    print "precision is",metrics.precision_score(y_true, y_score2)
    print "recall is",metrics.recall_score(y_true, y_score2)
    print "F1 is",metrics.f1_score(y_true, y_score2)
    print "accuracy is",metrics.accuracy_score(y_true, y_score2)

    with open(file_path, 'r') as csvFile:
        csvreader = csv.reader(csvFile)
        for line in csvreader:
            if line


