# -*- coding: utf-8 -*-
from sklearn import metrics
import random
import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import csv


def parse_args():
    '''
	Parses the node2vec arguments.
	'''
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--output',
                        nargs='?',
                        default='resultat1.emb',
                        help='Embeddings path')
    parser.add_argument('--dimensions',
                        type=int,
                        default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length',
                        type=int,
                        default=50,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks',
                        type=int,
                        default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size',
                        type=int,
                        default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter',
                        default=1,
                        type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers',
                        type=int,
                        default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p',
                        type=float,
                        default=0.25,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q',
                        type=float,
                        default=0.5,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument(
        '--weighted',
        dest='weighted',
        action='store_true',
        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted',
                        dest='unweighted',
                        action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed',
                        dest='directed',
                        action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected',
                        dest='undirected',
                        action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def learn_embeddings(walks):
    '''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
    # walks = [['a'+ str(x) for x in walk] for walk in walks]
    walks = [[str(x) for x in walk] for walk in walks]
    # walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks,
                     size=args.dimensions,
                     window=args.window_size,
                     min_count=0,
                     sg=1,
                     workers=args.workers,
                     iter=args.iter)
    # model.save_word2vec_format(args.output)
    model.wv.save_word2vec_format(args.output)

    return


def main(args, Samplegraph):
    '''
	Pipeline for representational learning for all nodes in a graph.
	'''
    nx_G = Samplegraph
    print nx_G.number_of_nodes()
    print nx_G.edges(data=True)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks)


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
    num = int(G.number_of_edges() / 2)  # 取样数目
    positive_sample = random.sample(G.edges(), num)  # 从list中随机获取5个元素，作为一个片断返回
    neg_sample = random.sample(list(nx.non_edges(G)), num)
    for u, v in positive_sample:
        G.remove_edge(u, v)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    tagetlist = [positive_sample, neg_sample, num, G]
    return tagetlist


def print_metrics(samlist):
    result = open("soc_wiki.csv", "ab+")
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
    print "AUC is", metrics.roc_auc_score(y_true, y_score)
    threshold = np.median(y_score)
    y_score2 = [int(item > threshold) for item in y_score]
    print "precision is", metrics.precision_score(y_true, y_score2)
    print "recall is", metrics.recall_score(y_true, y_score2)
    print "F1 is", metrics.f1_score(y_true, y_score2)
    print "accuracy is", metrics.accuracy_score(y_true, y_score2)
    writer.writerow([
        metrics.roc_auc_score(y_true, y_score),
        metrics.precision_score(y_true, y_score2),
        metrics.recall_score(y_true, y_score2),
        metrics.f1_score(y_true, y_score2),
        metrics.accuracy_score(y_true, y_score2)
    ])


if __name__ == "__main__":
    i = 0
    while i < 1:
        tmplist = create_sample('.\\Dataset\\bio-HS-HT.edges')
        tarG = tmplist[3]
        args = parse_args()
        main(args, tarG)
        print_metrics(tmplist)
        i = i + 1
