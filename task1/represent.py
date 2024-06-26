import abc_py
import numpy as np
import torch
import os
import pickle
import shutil

def represent(state, label):
    _abc = abc_py.AbcInterface()
    _abc.start()
    _abc.read(state)
    data = {}
    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype = int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype = int)
    edge_src_index = []
    edge_target_index = []
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype = torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['nodes'] = numNodes
    data['label'] = label
    return data

for filename in os.listdir('currentAIG2/train'):
    if filename.endswith('.aig'):
        state = 'currentAIG2/train/' + filename
        with open('currentAIG2/train_label/' + filename.replace('.aig', '.pkl'), 'rb') as f:
            label = pickle.load(f)
        data = represent(state, label)
        with open('dataset2/train/' + filename.replace('.aig', '.pkl'), 'wb') as f:
            pickle.dump(data, f)

for filename in os.listdir('currentAIG2/test'):
    if filename.endswith('.aig'):
        state = 'currentAIG2/test/' + filename
        with open('currentAIG2/test_label/' + filename.replace('.aig', '.pkl'), 'rb') as f:
            label = pickle.load(f)
        data = represent(state, label)
        with open('dataset2/test/' + filename.replace('.aig', '.pkl'), 'wb') as f:
            pickle.dump(data, f)