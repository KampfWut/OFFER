#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import csv
from sklearn import metrics
import networkx as nx
from io import open
import numpy as np
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
import graph
import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram
from six import text_type as unicode
from six import iteritems
from six.moves import range
import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


def process(args,graphlist):

  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)
  elif args.format == "edges":
    G = graph.load_edgelist(graphlist, undirected=args.undirected)
    #G = graph.load_edgelist(args.input, undirected=args.undirected)
  elif args.format == "mat":
    G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edges', 'mat'" % args.format)

  print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.number_walks

  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  print("Data size (walks*length): {}".format(data_size))

  if data_size < args.max_memory_data_size:
    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
    print("Training...")
    model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
  else:
    print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
    print("Walking...")

    walks_filebase = args.output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                         path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
                                         num_workers=args.workers)

    print("Counting vertex frequency...")
    if not args.vertex_freq_degree:
      vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
    else:
      # use degree distribution for frequency in tree
      vertex_counts = G.degree(nodes=G.iterkeys())

    print("Training...")
    walks_corpus = serialized_walks.WalksCorpus(walk_files)
    model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                     size=args.representation_size,
                     window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)

  model.wv.save_word2vec_format(args.output)


def main(Graph):
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='edges',
                      help='File format of input file')

  parser.add_argument('--input', nargs='?', default='E:\code\Motif\dataset\soc-hamsterster.edges',
                      help='Input graph file')

  parser.add_argument('--output', default='resultat1.emb',
                      help='Output representation file')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--representation-size', default=2, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=50, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=10, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=8, type=int,
                      help='Number of parallel processes.')


  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  process(args,Graph)


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
        return round(dot_product / ((normA ** 0.5) * (normB ** 0.5)) * 100, 2)

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

#创建样本
def create_sample(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as csvFile:
        for line in csvFile:
            node1 = str(line[:-1].split(',')[0])
            node2 = str(line[:-1].split(',')[1])
            G.add_edge(node1, node2)
    num = int(G.number_of_edges() / 2)  # 取样数目
    positive_sample = random.sample(G.edges(), num)  # 从list中随机获取5个元素，作为一个片断返回
    neg_sample = random.sample(list(nx.non_edges(G)), num)
    for u, v in positive_sample:
        G.remove_edge(u, v)
    edgedic ={}
    for a in G.nodes():
        edgedic[a] = list(G.neighbors(a))
    tagetlist = [positive_sample, neg_sample, num,G]
    return tagetlist

#评价函数求AUC等评价函数
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
    writer.writerow([metrics.roc_auc_score(y_true, y_score), metrics.precision_score(y_true, y_score2),
                     metrics.recall_score(y_true, y_score2),
                     metrics.f1_score(y_true, y_score2), metrics.accuracy_score(y_true, y_score2)])

if __name__ == "__main__":
    i = 0
    while i < 1:
        tmplist = create_sample('karate_edge_list.csv')
        tarG = tmplist[3]
        main(tarG)
        print_metrics(tmplist)
        i = i + 1
        print i

