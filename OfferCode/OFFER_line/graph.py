"""Graph utilities."""

# from time import time
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp

__author__ = "Zhang Zhengyan"
__email__ = "zhangzhengyan14@mails.tsinghua.edu.cn"


class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_edgelist(self, graph):
        self.G = nx.Graph()
        for edge in graph.edges(data=True):
            src = edge[0]
            dst = edge[1]
            w = edge[2]['weight']
            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)
            self.G[src][dst]['weight'] = float(w)
            self.G[dst][src]['weight'] = float(w)
        self.G.add_nodes_from(graph.nodes())
        print(self.G.number_of_nodes())
        self.encode_node()

    def read_node_label(self, filename):
        nodedic = {}
        with open(filename, 'r') as csvFile:
            for line in csvFile:
                node = str(line[:-1].split(',')[0])
                weight = float(line[:-1].split(',')[1])
                nodedic[node] = weight
        return nodedic
