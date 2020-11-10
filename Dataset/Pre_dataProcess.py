# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
import codecs

result = codecs.open('inf-openflights1.edges', 'w', 'utf-8')
with open('inf-openflights.edges', 'r') as csvFile:
    for line in csvFile:
        node1 = line.split(' ')[0]
        node2 = line.split(' ')[1]
        node3 = line.split(' ')[2]
        result.write(node1 + '\t' + node2 + node3)
result.close()
