# -*- coding: utf-8 -*-
import sys
import codecs
import json
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


def three(G):
    edge_list = []
    node_list = []
    i = 0
    for cycle in nx.enumerate_all_cliques(G):
        if len(cycle) == 3:
            for n in cycle:
                node_list.append(n)
            temp_co_list = combinations(cycle, 2)  # 添加作者列表
            for temp in temp_co_list:
                edge_list.append(temp)

    print 'first step'

    with open('b3_edge_result.csv', 'wb') as csvFile:
        csv_writer = csv.writer(csvFile)
        for a in G.edges():
            if a in set(edge_list):
                w = 1 + float(edge_list.count(a)) / 3
                csv_writer.writerow([a[0], a[1], w])
            else:
                csv_writer.writerow([a[0], a[1], 1])
    print 'edge step'

    with open('b3_node_result.csv', 'wb') as csvFile:
        csv_writer = csv.writer(csvFile)
        for a in G.nodes():
            if a in set(node_list):
                w = 1 + float(node_list.count(a)) / 3
                csv_writer.writerow([a, w])
            else:
                csv_writer.writerow([a, 1])
    print 'node step'


if __name__ == "__main__":
    G = nx.Graph()
    print 'building graph...'
    with open(DATA_PATH + 'bio-HS-HT.edges', 'r') as csvFile:
        for line in csvFile:
            node1 = str(line[:-1].split(' ')[0])
            node2 = str(line[:-1].split(' ')[1])
            G.add_edge(node1, node2)
    print 'nodes number:', G.number_of_nodes()
    print 'edges number', G.number_of_edges()
    three(G)
