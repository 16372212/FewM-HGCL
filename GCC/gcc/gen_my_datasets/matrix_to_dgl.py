#!/usr/bin/env python
# encoding: utf-8
# File Name: matrix_to_dgl.py
# Author: Zhen Ziyang
# Create Time: 2021/11/6 21:50
# TODO:

import io
import itertools
import os
import os.path as osp
from collections import defaultdict, namedtuple
import pickle
import dgl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from dgl.data.tu import TUDataset
from scipy.sparse import linalg
import logging
import sys
root_path = os.path.abspath("./")
sys.path.append(root_path)

from gcc.Sample import Sample, Node


GRAPH_OUTPUT_PATH = 'gcc/gen_my_datasets/graph.pkl'
NODE_OUTPUT_PATH = 'gcc/gen_my_datasets/nodes.pkl'
SAMPLE_LIST_OUTPUT_PATH = 'gcc/gen_my_datasets/sample_list.pkl'
API_MATRIX_OUTPUT_PATH = 'gcc/gen_my_datasets/api_matrix1_3.pkl'
API_INDEX_OUTPUT_PATH = 'gcc/gen_my_datasets/api_index_map.pkl'


graph = {}
Nodes = []
sample_list = []
label_index = {}
labels = []
API_LIST_LEN = 32


with open(GRAPH_OUTPUT_PATH, 'rb') as fr:
    graph = pickle.load(fr)

with open(NODE_OUTPUT_PATH, 'rb') as f:
    Nodes = pickle.load(f)

with open(SAMPLE_LIST_OUTPUT_PATH, 'rb') as f:
    sample_list = pickle.load(f)


with open(API_MATRIX_OUTPUT_PATH, 'rb') as f:
    api_matrix = pickle.load(f)


with open(API_INDEX_OUTPUT_PATH, 'rb') as f:
    api_index_map = pickle.load(f)


def get_label_index(label):
    if label not in label_index:
        label_index[label] = len(labels)
        labels.append(label)
    return label_index[label]


def get_api_property(type_, api_name):
    # print(api_index_map) # name:index
    
    if type_ == 'api' and api_name in api_index_map:
        api_array = api_matrix[api_index_map[api_name]]
        return api_array
    return [int(0)]*API_LIST_LEN


def gen_normal_dgl(sample, type_dict):
    node_id = [sample.num]
    node_type_feat = [4]
    api_pro_feat = [[int(0)]*API_LIST_LEN]

    td = {'api':0, 'network':1, 'sign':2, 'file':3, 'process':4}

    for type_ in type_dict:
        if type_ not in td:
            continue
        for n in type_dict[type_]:
            # 节点的id
            node_id.append(n)
            # 节点的type
            node_type_feat.append(td[type_])
            # 节点如果是api，则添加20维度的api属性向量. 如果不是则添加32维度的0向量
            api_pro_arr = get_api_property(type_, Nodes[n].name)
            api_pro_feat.append(api_pro_arr)
            # 构建相对应的q：api属性的遮掩
    G = dgl.DGLGraph()
    G.add_nodes(len(node_id))
    G.add_edges(0, list(range(1,len(node_id))))
    G.ndata['node_id'] = torch.tensor(node_id)
    G.ndata['node_type'] = torch.tensor(node_type_feat)
    G.ndata['api_pro'] = torch.tensor(api_pro_feat)
    # print(G.ndata['api_pro'].shape)
    return G


def gen_one_dgl_from_dgl(sample, type_dict):
    """这里可能需要将sample专门设置成process的一种？"""
    node_id = []
    node_type_feat = []


    td = {'api':0, 'network':1, 'sign':2, 'file':3, 'process':4}

    for type_ in type_dict:
        if type_ not in td:
            continue
        for n in type_dict[type_]:
            node_id.append(n)
            node_type_feat.append(td[type_])

    u, v = torch.tensor([int(sample.num)]*len(node_id),dtype=torch.int64), torch.tensor(node_id,dtype=torch.int64)
    g = dgl.graph((u, v))
    
    # 这里node_type没有添加
    
    # ******************************
    # api属性遮掩 0.4的可能进行遮掩  待增加一个api的遮掩属性
    # 某些节点需要增加或减少，需要增加一个road属性
    # ******************************
    return g


def samples_to_dgl():
    graph_lists = []
    label_lists = []
    label_num_dict = {} # label_index : nums
    label_num_threshold = 120

    for sample_num in graph:
        # dgl create
        if sample_num[0] == 's' and int(sample_num[1:]) < len(sample_list):
            sam = sample_list[int(sample_num[1:])]
            label = get_label_index(sam.family)
            # 判断这个类型的数目，选择性添加sample
            if label in label_num_dict:
                if(label_num_dict[label] > label_num_threshold): continue 
                label_num_dict[label] += 1
            else:
                label_num_dict[label] = 1

            label_lists.append(label)
            g = gen_normal_dgl(sam, graph[sample_num])
            graph_lists.append(g)
    return graph_lists, label_lists


samples_to_dgl()