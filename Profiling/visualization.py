# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import manifold
import networkx as nx
import time
import matplotlib.pyplot as plt


def read_embedding(file_path):
    vector_dic = {}
    nodename_list = []
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
                nodename_list.append(node)
    return [vector_dic, nodename_list]


if __name__ == "__main__":

    print(">> Read EMB_Result...")
    [emb_result,
     nodename] = read_embedding("resultat1.emb")
    X = []
    for name in nodename:
        X.append(emb_result[name])

    print(">> Run T-SNE...")
    tsne = manifold.TSNE(n_components=2, init='pca')
    X_tsne = tsne.fit_transform(X)

    print(">> Draw...")
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.plot(X_norm[i, 0], X_norm[i, 1], 'rx')
    plt.xticks([])
    plt.yticks([])
    plt.show()