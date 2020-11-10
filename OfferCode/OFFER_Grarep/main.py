# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from graph import *
import networkx as nx
from grarep import GraRep
import time


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input',
                        help='Input graph file',
                        default='soc-wiki-Vote.edges')
    parser.add_argument('--output',
                        help='Output representation file',
                        default='resultat1.emb')
    parser.add_argument(
        '--representation-size',
        default=128,
        type=int,
        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--method',
                        default='grarep',
                        help='The learning method')
    parser.add_argument('--graph-format',
                        default='edgelist',
                        help='Input graph format')
    parser.add_argument('--weighted',
                        action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--kstep',
                        default=4,
                        type=int,
                        help='Use k-step transition probability matrix')
    args = parser.parse_args()
    return args


def main(args, samplegraph):
    t1 = time.time()
    g = Graph()
    print("Reading...")
    g.read_edgelist(samplegraph)
    model = GraRep(graph=g, Kstep=args.kstep, dim=args.representation_size)
    t2 = time.time()
    print(t2 - t1)
    print("Saving embeddings...")
    model.save_embeddings(args.output)


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
            node1 = str(line[:-1].split(',')[0])
            node2 = str(line[:-1].split(',')[1])
            G.add_edge(node1, node2)
            G[node1][node2]['weight'] = float(line[:-1].split(',')[2])
    num = int(G.number_of_edges() / 2)  # 取样数目
    positive_sample = random.sample(G.edges(), num)  # 从list中随机获取一半的边作为一个片断返回
    neg_sample = random.sample(list(nx.non_edges(G)), num)
    for u, v in positive_sample:
        G.remove_edge(u, v)
    tagetlist = [positive_sample, neg_sample, num, G]
    return tagetlist


def print_metrics(samlist):
    result = open('new_soc_wiki.csv', 'ab+')
    writer = csv.writer(result)
    emb_path = 'resultat1.emb'
    num = samlist[2]
    positive_sample = samlist[0]
    neg_sample = samlist[1]
    pos_list = [1] * num
    pos_list.extend([0] * num)
    y_true = np.array(pos_list)
    y_scores = []
    vvdict = read_embedding(emb_path)
    for i in positive_sample:
        y_scores.append(cosine_similarity(vvdict.get(i[0]), vvdict.get(i[1])))
    for i in neg_sample:
        y_scores.append(cosine_similarity(vvdict.get(i[0]), vvdict.get(i[1])))
    y_score = np.array(y_scores)
    print("AUC is", metrics.roc_auc_score(y_true, y_score))
    threshold = np.median(y_score)
    y_score2 = [int(item > threshold) for item in y_score]
    print("precision is", metrics.precision_score(y_true, y_score2))
    print("recall is", metrics.recall_score(y_true, y_score2))
    print("F1 is", metrics.f1_score(y_true, y_score2))
    print("accuracy is", metrics.accuracy_score(y_true, y_score2))
    writer.writerow([
        float(metrics.roc_auc_score(y_true, y_score)),
        float(metrics.precision_score(y_true, y_score2)),
        float(metrics.recall_score(y_true, y_score2)),
        float(metrics.f1_score(y_true, y_score2)),
        float(metrics.accuracy_score(y_true, y_score2))
    ])


if __name__ == "__main__":
    i = 0
    while i < 1:
        tmplist = create_sample('.\\Dataset\\3_edge_result.csv')
        tarG = tmplist[3]
        print(len(tarG.edges()), tarG.edges())
        print(tarG.number_of_nodes())
        args = parse_args()
        main(args, tarG)
        print_metrics(tmplist)
        i = i + 1
