# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
from sklearn import manifold, datasets


class Spectral(object):
    def __init__(self, graph, dim):
        self.g = graph
        self.dim = int(dim)
        self.train()

    def getAdjMat(self, nodedic_file):
        graph = self.g.G
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        node_dic = self.g.read_node_label(nodedic_file)
        adj = np.ones((node_size, node_size))
        for edge in self.g.G.edges(data=True):
            adj[look_up[edge[0]]][look_up[edge[1]]] = float(edge[2]['weight'])
            adj[look_up[edge[1]]][look_up[edge[0]]] = float(edge[2]['weight'])
        # ScaleSimMat
        for node in self.g.G.nodes():
            adj[look_up[node]][look_up[node]] = node_dic.get(node)
        ttemp = np.sum(adj, axis=1)
        return np.matrix(adj / ttemp)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

    def train(self):
        self.adj = self.getAdjMat('.\\Dataset\\3_node_result.csv')
        self.node_size = self.adj.shape[0]
        print("Computing embedding")
        self.SpeMat = manifold.SpectralEmbedding(
            n_components=self.dim).fit_transform(self.adj)
        # get embeddings
        self.vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self.SpeMat):
            self.vectors[look_back[i]] = embedding