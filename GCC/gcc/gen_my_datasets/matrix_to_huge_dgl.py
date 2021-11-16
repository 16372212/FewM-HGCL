#!/usr/bin/env python
# encoding: utf-8
# File Name: matrix_to_dgl.py
# Author: Zhen Ziyang
# Create Time: 2021/11/10 21:50
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
import copy
root_path = os.path.abspath("./")
sys.path.append(root_path)

from gcc.Sample import Sample, Node


GRAPH_OUTPUT_PATH = 'gcc/gen_my_datasets/graph.pkl'
NODE_OUTPUT_PATH = 'gcc/gen_my_datasets/nodes.pkl'
SAMPLE_LIST_OUTPUT_PATH = 'gcc/gen_my_datasets/sample_list.pkl'
API_MATRIX_OUTPUT_PATH = 'gcc/gen_my_datasets/api_matrix1_3.pkl'
API_INDEX_OUTPUT_PATH = 'gcc/gen_my_datasets/api_index_map.pkl'
SAMPLE_NUM_TO_NODE_ID_PATH = 'gcc/gen_my_datasets/sample_num_to_node_id.pkl'

GRAPH_INPUT_PATH = 'gcc/gen_my_datasets/subgraphs_train_data.pkl'


graph = {}
Nodes = []
sample_list = []
label_index = {}
labels = []
left_node = []
right_node = []
API_LIST_LEN = 32
td = {'api':0, 'network':1, 'sign':2, 'file':3, 'process':4}

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

with open(SAMPLE_NUM_TO_NODE_ID_PATH, 'rb') as f:
    sample_num_to_node_id = pickle.load(f)

def get_label_index(label):
    if label not in label_index:
        label_index[label] = len(labels)
        labels.append(label)
    return label_index[label]


def gen_node_relation(sam, type_dict):
    node_id = sample_num_to_node_id[sam.num]
    # 加一些边
    for type_ in type_dict:
        for num in type_dict[type_]:
            left_node.append(node_id)
            right_node.append(num)

            left_node.append(num)
            right_node.append(node_id)


def get_api_property(type_, api_name):
    if type_ == 'api' and api_name in api_index_map:
        api_array = api_matrix[api_index_map[api_name]]
        return api_array
    return [int(0)]*API_LIST_LEN

    
def gen_ndata_property():
    node_type = [] # node_id: node_type
    api_pro = [] # node_id: api_pro 
    
    for node in Nodes:
        node_type.append(td[node.type_])
        api_pro.append(get_api_property(node.type_, node.name))
    return node_type, api_pro  


def samples_to_dgl():
    sample_id_lists = []
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
            
            sample_id_lists.append(sample_num_to_node_id[sam.num])
            label_lists.append(label)
            # 构造left_node, right_node
            gen_node_relation(sam, graph[sample_num])

    # 构建dgl_graph
    dgl_graph = dgl.DGLGraph((torch.tensor(left_node), torch.tensor(right_node)))
    node_type, api_pro = gen_ndata_property()
    dgl_graph.ndata['node_type'] = torch.tensor(node_type)
    dgl_graph.ndata['api_pro'] = torch.tensor(api_pro)
    dgl_graph.ndata['node_id'] = torch.tensor([*range(0, len(node_type), 1)])
    return dgl_graph, sample_id_lists, label_lists



def get_aug_of_graph_list(g, sample_id, len2_node, len3_node):
    """得到给出的子图的所有增强子图，并返回list"""
    aug_list = []
    # api属性遮掩
    tuple_temp = (torch.tensor([sample_id]), len2_node, len3_node)
    subv = torch.unique(torch.cat(tuple_temp)).tolist()
    api_aug_g = g.subgraph(subv)
    api_aug_g.copy_from_parent()
    mask_k = torch.zeros(len(api_aug_g.ndata['api_pro']), dtype=torch.int64).bernoulli(0.6).reshape((len(api_aug_g.ndata['api_pro']),1))
    api_aug_g.ndata['api_pro'] = api_aug_g.ndata['api_pro'] * mask_k
    aug_list.append(api_aug_g)
    # 节点减少与边的增加
    for i in range(0, len(td)):
        for j in range(i+1, len(td)):
            delete_type = [i, j]
            
            # 减少节点, 通过判断第二层的节点是否有一个以上的节点相连
            del_temp_subv = copy.copy(subv) 
            for node_id in len2_node:
                # 属于被删除的类型
                if g.ndata['node_type'][node_id] in delete_type:
                    # 并且入度为1
                    # 问题： 发现二层的节点几乎都会和多个sample相连
                    # print(f' the len of that is {len(dgl.bfs_nodes_generator(g, node_id)[0])}')
                    # print(f' the len of that is {len(dgl.bfs_nodes_generator(g, node_id)[1])}')
                    # print(f' the len of that is {len(dgl.bfs_nodes_generator(g, node_id)[2])}')
                    # print()
                    if len(dgl.bfs_nodes_generator(g, node_id)[1]) == 1:
                        if node_id in del_temp_subv: # 这里为什么会不存在呢
                            del_temp_subv.remove(node_id) # 从list中移除一个节点
                        else:
                            logging.warn(f'node {node_id} 竟然不存在, 怀疑发生了浅拷贝')
            temp_subgraph = g.subgraph(del_temp_subv)
            temp_subgraph.copy_from_parent()
            aug_list.append(temp_subgraph)
            
            # 增加边。遍历第三层的节点，都直接与sample相连
            temp_g = copy.copy(g)
            for node_id in len3_node:
                if g.ndata['node_type'][node_id] in delete_type:
                    # 增加一个这样的边
                    # 问题： 发现三层的节点太多了
                    temp_g.add_edges(torch.tensor([node_id, sample_id]), torch.tensor([sample_id, node_id]))

            temp_subgraph = temp_g.subgraph(subv)
            temp_subgraph.copy_from_parent()
            aug_list.append(temp_subgraph)
    return aug_list


def get_each_sample_subgraph():
    g, sample_id_lists, label_lists = samples_to_dgl()
    graph_k_list = []
    graph_k_aug_list = []
    print('sample id的总数：')
    print(len(sample_id_lists))
    for sample_id in sample_id_lists:
        #2倍的所有邻居构图

        len1_node, len2_node, len3_node = dgl.bfs_nodes_generator(g, sample_id)[:3]
        
        print(f'sample id:{sample_id}, node num:{len(len3_node)}')
        tuple_temp = (len1_node, len2_node, len3_node)
        subv = torch.unique(torch.cat(tuple_temp)).tolist()
        subg = g.subgraph(subv)
        # subg.copy_from_parent()
        # 问题：检查距离为2的邻居的类型，检查结果：全是node4。出现的问题：process类型节点过多。是否考虑将process与sample完全分开？
        temp_list = get_aug_of_graph_list(g, sample_id, len2_node, len3_node)
        graph_k_aug_list += temp_list
        graph_k_list.append(subg)
    print(f'finished')

    # save it 
    with open(GRAPH_INPUT_PATH, 'wb') as f1:
        pickle.dump(graph_k_list, f1)
        pickle.dump(label_lists, f1)
        pickle.dump(graph_k_aug_list, f1)
    

    return graph_k_list, label_lists, graph_k_aug_list


def get_saved_sample_subgraph():
    with open(GRAPH_INPUT_PATH, 'rb') as f:
        graph_k_list = pickle.load(f)
        label_lists = pickle.load(f)
        graph_k_aug_list = pickle.load(f)

    return graph_k_list, label_lists, graph_k_aug_list 


get_each_sample_subgraph()
# get_saved_sample_subgraph()