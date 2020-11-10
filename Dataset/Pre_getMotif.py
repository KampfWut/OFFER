# -*- coding: utf-8 -*-
import sys
import codecs
import itertools
import random
from math import *
import time
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from itertools import product
import networkx as nx
from itertools import combinations
import collections
import numpy as np
import time
import codecs
import csv
import sys
#改变编码格式
reload(sys)
sys.setdefaultencoding('utf8')

DATA_PATH = 'E:\\OfflineCode\\OFFER\\Dataset\\'


def quchong(slist):
    new_slist = []
    for id in slist:
        if id not in new_slist:
            new_slist.append(id)
    return new_slist


def get_result(G):
    s_list = []
    edge_list = []
    for x in G.edges():
        n_list = list(nx.common_neighbors(G, x[0], x[1]))
        if len(n_list) > 0:
            tmp_edge = [x[0], x[1], len(n_list)]
            edge_list.append(tmp_edge)
            for y in n_list:
                tmplist = [x[0], x[1], y]
                s_list.append(sorted(tmplist))
        else:
            tmp_edge = [x[0], x[1], 1]
            edge_list.append(tmp_edge)
    print 'edge step'

    new_slist = quchong(s_list)
    node_list = []
    for trlist in new_slist:
        for node in trlist:
            node_list.append(node)
    print 'node step'

    with open('3_edge_result.csv', 'wb') as csvFile:
        csv_writer = csv.writer(csvFile)
        for a in edge_list:
            w = 1 + float(a[2]) / 3
            csv_writer.writerow([a[0], a[1], w])

    with open('3_node_result.csv', 'wb') as csvFile:
        csv_writer = csv.writer(csvFile)
        for a in G.nodes():
            if a in node_list:
                w = 1 + float(node_list.count(a)) / 3
                csv_writer.writerow([a, w])
            else:
                csv_writer.writerow([a, 1])


if __name__ == "__main__":
    G = nx.Graph()
    print 'building graph...'
    with open(DATA_PATH + 'rt-twitter-copen.edges', 'r') as csvFile:
        for line in csvFile:
            #print line
            node1 = str(line[:-1].split('\t')[0])
            #print "node1: " + str(node1)
            node2 = str(line[:-1].split('\t')[1])
            #print "node2: " + str(node2)
            G.add_edge(node1, node2)
    print 'nodes number:', G.number_of_nodes()
    print 'edges number', G.number_of_edges()
    get_result(G)
    # n = G.number_of_nodes()
    # p =  G.number_of_edges()
    # c = four_1(G)
    # print 'four is ok'
    # a = three_1(G)
    # print 'three is ok'
    # b = jizhuazi(G)
    # print 'jizhuazi is ok'
    #
    # A_Graph = [a[0], a[1], b, c[0], c[1], c[2]]
    # print A_Graph
    # motif_sanjiao = []
    # motif_sanjiao1 = []
    # motif_jizhuazi = []
    # motif_fang = []
    # motif_fang1= []
    # motif_fang2 = []
    # i = 0
    # for i in range(0,10):
    #     G1 = nx.gnm_random_graph(n, p)
    #     c = four_1(G1)
    #     a = three_1(G1)
    #     b = jizhuazi(G1)
    #     motif_sanjiao.append(a[0])
    #     motif_sanjiao1.append(a[1])
    #     motif_jizhuazi.append(b)
    #     motif_fang.append(c[0])
    #     motif_fang1.append(c[1])
    #     motif_fang2.append(c[2])
    #     print i
    #     i = i+1
    #
    # print 'random 10 times..'
    # Z_sanjiao = (A_Graph[0] - np.average(motif_sanjiao)+1)/(np.std(motif_sanjiao,ddof=1)+6)
    # Z_sanjiao1 = (A_Graph[1] - np.average(motif_sanjiao1)+1)/(np.std(motif_sanjiao1,ddof=1)+6)
    # Z_jizhuazi = (A_Graph[2] - np.average(motif_jizhuazi)+1)/(np.std(motif_jizhuazi,ddof=1)+6)
    # Z_fang = (A_Graph[3] - np.average(motif_fang)+1)/(np.std(motif_fang,ddof=1)+6)
    # Z_fang1 = (A_Graph[4] - np.average(motif_fang1)+1)/(np.std(motif_fang1,ddof=1)+6)
    # Z_fang2 = (A_Graph[5] - np.average(motif_fang2)+1)/(np.std(motif_fang2,ddof=1)+6)
    # Zlist = [Z_sanjiao,Z_sanjiao1,Z_jizhuazi,Z_fang,Z_fang1,Z_fang2]
    # print Zlist
