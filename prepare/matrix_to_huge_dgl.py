#!/usr/bin/env python
# encoding: utf-8
# File Name: matrix_to_dgl.py
# Author: Zhen Ziyang
# Create Time: 2021/11/10 21:50
# 该py文件主要干两件事：构造dgl子图， 构造dgl增强子图


import os
import pickle

import dgl
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')
import networkx as nx
import numpy as np
import torch
import logging
import sys
import copy
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
from typing import List

from Sample import Node, Sample

root_path = os.path.abspath("./")
sys.path.append(root_path)

GRAPH_OUTPUT_PATH = 'mid_data/graph.pkl'
NODE_OUTPUT_PATH = 'mid_data/nodes.pkl'
SAMPLE_LIST_OUTPUT_PATH = 'mid_data/sample_list.pkl'
API_MATRIX_OUTPUT_PATH = 'mid_data/api_matrix1_3.pkl'
API_INDEX_OUTPUT_PATH = 'mid_data/api_index_map.pkl'
SAMPLE_NUM_TO_NODE_ID_PATH = 'mid_data/sample_num_to_node_id.pkl'

DGL_OUTPUT_PATH = 'mid_data/gcc_input/subgraphs_train_data.bin'  # 构造的dgl

GRAPH_SUB_AUG_INPUT_PATH = 'mid_data/gcc_input/aug_graphs_10/'  # 构造的正样本的存放路径

# 前置数据（已有，直接读取）
graph = {}  # 二维矩阵dict{}。从原始数据集通过dfs构造出来的二维matrix
Nodes: List[Node] = []  # 记录矩阵中所有节点的list。Node中的index和graph中节点的index是对应关系。相当于matrix表示的是节点之间的关系，nodes列出所有节点及其属性。
sample_list: List[Sample] = []  # 计算出来的存储所有sample的列表

# 中间数据
label_index = {}  # Dict[大类label名称: labels中的index]
labels = []  # List[大类label名称]，二者对应关系

big_label_index = {}  # Dict[family label名称: labels中的index]
big_labels = []  # List[family label名称]，二者对应关系

left_nodes = []  # 构造dgl用到的中间数据。dgl中的边的起始节点
right_nodes = []  # dgl中的边的终止节点

API_LIST_LEN = 32
td = {'api': 0, 'network': 1, 'reg': 2, 'file': 3, 'process': 4}

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

# 这个是sample num->node id的对应关系
with open(SAMPLE_NUM_TO_NODE_ID_PATH, 'rb') as f:
    sample_num_to_node_id = pickle.load(f)


def get_label_index(label):
    if label not in label_index:
        label_index[label] = len(labels)
        labels.append(label)
    return label_index[label]


def get_big_label_index(label):
    if label not in big_label_index:
        big_label_index[label] = len(big_labels)
        big_labels.append(label)
    return big_label_index[label]


def gen_node_relation(node_id, type_dict):
    # 加一些边, 单向
    for type_ in type_dict:
        for num in type_dict[type_]:
            left_nodes.append(node_id)
            right_nodes.append(num)
            # 下面注释若打开说明图是双向边
            # left_node.append(num)
            # right_node.append(node_id)


def get_api_property(type_, api_name):
    if type_ == 'api' and api_name in api_index_map:
        api_array = api_matrix[api_index_map[api_name]]
        return api_array
    return [int(0)] * API_LIST_LEN


def gen_ndata_property():
    node_type = []  # node_id: node_type
    api_pro = []  # node_id: api_pro
    for node in Nodes:
        node_type.append(td[node.type_])
        api_pro.append(get_api_property(node.type_, node.name))
    return node_type, api_pro


def draw_dgl_from_matrix():
    """根据二维矩阵graph和节点属性列表nodes来构建dgl的函数"""
    sample_id_lists = []  # sample列表，记录用来构建dgl的所有sample的index
    big_label_lists = []  # label列表（大类）。Trojan, Backdoor之类的
    family_label_lists = []  # label列表（小类) 大类中的家族名称。每个大类label下会包含各个family
    error_sample_nodes_in_graph = 0
    print(
        f'===============sample num:{len(sample_list)}===total num{len(Nodes)} , graph len:{len(graph)}:====================')

    for sample_num in graph:
        # 遍历二维矩阵中node节点之间的关系构建dgl
        node_id = -1
        if sample_num[0] != 's':
            # graph中该节点没有以s开头，说明该节点不可以作为一个sample。因此只是加入到node_id中
            node_id = int(sample_num)
            error_sample_nodes_in_graph += 1
        if sample_num[0] == 's' and int(sample_num[1:]) < len(sample_list):
            # 以s开头，属于合理的sample，开始构造single subgraph
            sample_num_int = int(sample_num[1:])
            sam = sample_list[sample_num_int]
            node_id = sample_num_to_node_id[sam.num]
            big_label = get_big_label_index(sam.label)
            family_label = get_label_index(sam.family)

            sample_id_lists.append(node_id)
            big_label_lists.append(big_label)
            family_label_lists.append(family_label)

        # 记录graph二维矩阵中的节点之间的关系，生成left_nodes, right_nodes。
        # node_id是当前节点index值。graph[sample_num]: 代表和当前节点相关联的节点的列表
        gen_node_relation(node_id, graph[sample_num])

    print(
        f'======after creating : sample num:{len(sample_id_lists)}===label num{len(family_label_lists)}===big label num{len(big_label_lists)}=================')
    print(f'graph中有{len(graph) - len(family_label_lists)}个process节点')
    print(f'{error_sample_nodes_in_graph} nodes not begin with s in graph(file-process)、(reg-process)')
    # 构建dgl_graph
    dgl_graph = dgl.DGLGraph((torch.tensor(left_nodes), torch.tensor(right_nodes)))
    node_type, api_pro = gen_ndata_property()  # 根据Nodes的顺序得到的，未筛选
    # 构造dgl节点的三种属性
    dgl_graph.ndata['node_type'] = torch.tensor(node_type)
    dgl_graph.ndata['api_pro'] = torch.tensor(np.array(api_pro))
    dgl_graph.ndata['node_id'] = torch.tensor([*range(0, len(node_type), 1)])
    return dgl_graph, sample_id_lists, family_label_lists, big_label_lists


def ana_process_nodes(sample_id_lists):
    """通过查看这些数据的分布。发现file里面的节点不均匀，少数达到两千多。其他的节点个数都很正常"""
    i = 0
    for node in Nodes:
        if node.type_ == 'process' and node.num not in sample_id_lists:
            i += 1
    print(f'一共 {i} 个process节点，但是可能包含没有用到的process')
    dict = {}
    for sample_num in graph:
        for type_ in graph[sample_num]:
            if type_ not in dict:
                dict[type_] = [len(graph[sample_num][type_])]
            else:
                dict[type_].append(len(graph[sample_num][type_]))
    for type_ in dict:
        print(type_)
        dict[type_].sort()
        print(dict[type_])
        print()


def analyze_nodes(sample_id, len2_node, g):
    k = 0
    for node_id in len2_node:
        id_ = int(node_id)
        type_ = g.ndata['node_type'][id_]
        if type_ == 4:
            k += 1
    print(f'{sample_id} has {k} processes')


def draw_aug_dgls(huge_graph, sample_id_lists, family_label_lists, big_label_lists):
    """ 根据每个dgl A 构造正样本A'
    不加处理会产生OOM，内存超过。一个是运算时间很慢，但更重要的是，数据量太大，将所有的节点变成都统一存储在一起，内存容量会暴增
    解决办法:
        1. 长度为2的节点做一个过滤，判断是否是自己的processTree里的
        2. 减少api参数长度: 将长度为32，删除多余参数变成长度为12
        3. 将aug的数据放到batch里进行训练
    """
    aug_to_k_index = []
    graph_k_list = []
    k_qnum_list = []
    print('sample id的总数：')
    print(len(sample_id_lists))
    index = 0
    # test num of nodes which have len3_node
    num_have_len3node = 0
    add_edge_num = 0
    del_node_num = 0
    k = 0
    for sample_id in sample_id_lists:
        k_qnum_list.append(len(aug_to_k_index))
        if len(dgl.bfs_nodes_generator(huge_graph, sample_id)) >= 3:
            num_have_len3node += 1
            len1_node, len2_node, len3_node = dgl.bfs_nodes_generator(huge_graph, sample_id)[:3]
            tuple_temp = (len1_node, len2_node, len3_node)
            an, dn = get_aug_of_graph_list(huge_graph, [], sample_id, len2_node, aug_to_k_index, len(graph_k_list),
                                           len3_node)
            add_edge_num += an
            del_node_num += dn
        else:
            len1_node, len2_node = dgl.bfs_nodes_generator(huge_graph, sample_id)[:2]
            tuple_temp = (len1_node, len2_node)
            an, dn = get_aug_of_graph_list(huge_graph, [], sample_id, len2_node, aug_to_k_index, len(graph_k_list))  # 应该是加其实
            add_edge_num += an
            del_node_num += dn
        subv = torch.unique(torch.cat(tuple_temp)).tolist()
        subg = huge_graph.subgraph(subv)
        subg.copy_from_parent()
        # draw_graph(subg, label_lists[index])
        graph_k_list.append(subg)
        index += 1
        k += 1
        if k % 100 == 0:
            print(f'{k} / {len(sample_id_lists)}')
    k_qnum_list.append(len(aug_to_k_index))
    print(f'------------add_edge_num: {add_edge_num}, del_node_num:{del_node_num}-------------')
    print(f'finished')
    print(f'{num_have_len3node} nodes which have len3node')
    print(len(aug_to_k_index))
    # save it
    graph_labels = {"glabel": torch.tensor(family_label_lists),
                    "big_label": torch.tensor(big_label_lists),
                    "k_q_index": torch.tensor(aug_to_k_index), "k_qnum": torch.tensor(k_qnum_list)}
    # graph_index = {"kindex": torch.tensor(aug_to_k_index)}
    save_graphs(DGL_OUTPUT_PATH, graph_k_list, graph_labels)
    return aug_to_k_index


def get_aug_of_graph_list(huge_graph, aug_list, sample_id, len2_node, aug_to_k_index, k_list_id, len3_node=None):
    """得到给出的子图的所有增强子图，并返回list"""
    # g = copy.deepcopy(huge_graph)
    g = huge_graph
    add_edge_num = 0
    del_node_num = 0
    # api属性遮掩, 遮掩两个
    tuple_temp = (torch.tensor([sample_id]), len2_node)
    if len3_node is not None:
        tuple_temp = (torch.tensor([sample_id]), len2_node, len3_node)
    subv = torch.unique(torch.cat(tuple_temp)).tolist()
    api_aug_g = g.subgraph(subv)
    api_aug_g.copy_from_parent()
    api_aug_g2 = copy.deepcopy(api_aug_g)
    mask_k = torch.zeros(len(api_aug_g.ndata['api_pro']), dtype=torch.int64).bernoulli(0.3).reshape(
        (len(api_aug_g.ndata['api_pro']), 1))
    mask_k2 = torch.zeros(len(api_aug_g.ndata['api_pro']), dtype=torch.int64).bernoulli(0.3).reshape(
        (len(api_aug_g.ndata['api_pro']), 1))
    api_aug_g.ndata['api_pro'] = api_aug_g.ndata['api_pro'] * mask_k
    api_aug_g2.ndata['api_pro'] = api_aug_g2.ndata['api_pro'] * mask_k2
    aug_list.append(api_aug_g)
    aug_list.append(api_aug_g2)
    aug_to_k_index.append(k_list_id)
    aug_to_k_index.append(k_list_id)

    if len3_node is None:
        # 还剩下1类型的没有删除，这里可以随机删除1类型的边
        del_temp_subv = []
        mask_node_index = torch.zeros(len(subv), dtype=bool).bernoulli(0.15)

        # 随机遮掩network类型的节点
        i = 0
        for tnode in subv:
            if mask_node_index[i]:
                del_temp_subv.append(tnode)

        temp_subgraph = g.subgraph(del_temp_subv)
        temp_subgraph.copy_from_parent()
        aug_list.append(temp_subgraph)
        aug_to_k_index.append(k_list_id)
        del_node_num += 1
        # save
        if not os.path.exists(GRAPH_SUB_AUG_INPUT_PATH):
            os.makedirs(GRAPH_SUB_AUG_INPUT_PATH)
        save_graphs(GRAPH_SUB_AUG_INPUT_PATH + 'aug_' + str(k_list_id) + '.bin', aug_list)
        # save_graphs(str(k_list_id)+'.bin', aug_list)
        return add_edge_num, del_node_num

    # 边的增强：因为只有两种类型的边：reg-process, file-process
    remove_td = [[2], [3], [4], [2, 4], [3, 4], [2, 3]]

    for delete_type_list in remove_td:
        del_temp_subv = copy.deepcopy(subv)  # 用来删除节点的
        temp_g = copy.deepcopy(g)  # 用来增加边的
        add_num = 0
        del_num = 0
        for node_id in len2_node:
            if g.ndata['node_type'][node_id] in delete_type_list:
                # 并且入度为0, 也就是这个节点没有再调用其他的节点了. 删除节点（删除节点的时候，不应该只和这两个节点有关系啊，只是增加节点的时候才有关系）
                if len(dgl.bfs_nodes_generator(g, node_id)) == 1:
                    if node_id in del_temp_subv:  # 这里遍历的二维节点一般会在subv中
                        del_temp_subv.remove(node_id)  # 从list中移除一个节点
                        del_num += 1
                    else:
                        logging.warning(f'node {node_id} 竟然不存在, 怀疑发生了浅拷贝')
                else:  # 二维节点遍历能得到相应的点，增加这个边
                    nodes_con_with_len2 = dgl.bfs_nodes_generator(g, node_id)[1]
                    if sample_id in nodes_con_with_len2:
                        logging.warning('二维节点竟然反向和sample相连接了')
                    add_num += 1
                    for node_id_of_3len in nodes_con_with_len2:
                        temp_g.add_edge(sample_id, int(node_id_of_3len))

        if del_num != 0:
            temp_subgraph = g.subgraph(del_temp_subv)
            temp_subgraph.copy_from_parent()
            aug_list.append(temp_subgraph)
            # print(f'node remove: {len(api_aug_g.nodes)} : {len(temp_subgraph.nodes)} / del_num={del_num}, subv:{len(subv)}, del_subv:{len(del_temp_subv)}')
            aug_to_k_index.append(k_list_id)
            del_node_num += 1
        if add_num != 0:
            temp_add_subgraph = temp_g.subgraph(subv)
            temp_add_subgraph.copy_from_parent()
            aug_list.append(temp_add_subgraph)
            # print(f'边的增加: {len(api_aug_g.edges)} : {len(temp_add_subgraph.edges)} / del_num={add_num} , g的边:{len(g.edges)}, temp_g的边:{len(temp_g.edges)}')
            aug_to_k_index.append(k_list_id)
            add_edge_num += 1
    # save
    save_graphs(GRAPH_SUB_AUG_INPUT_PATH + 'aug_' + str(k_list_id) + '.bin', aug_list)
    return add_edge_num, del_node_num


def draw_graph(g, label='normal'):
    # nxg = g.to_networkx(node_attrs=['n1'], edge_attrs=['e1'])
    nxg = dgl.to_networkx(g)
    fig, ax = plt.subplots()
    options = {
        "node_color": "#A0CBE2",
        "node_size": 40,
        "width": 4,
        "edge_cmap": plt.cm.Blues,
        "with_labels": False,
    }
    nx.draw(nxg, ax=ax, **options)
    ax.set_title('Class: {:d}'.format(label))
    plt.show()


def get_saved_sample_subgraph():
    graph_k_list, label_lists = load_graphs(DGL_OUTPUT_PATH)
    dataset = dict()
    dataset['graph_k_lists'] = graph_k_list  # 原本子图

    dataset['num_labels'] = len(label_lists['glabel'])
    dataset['graph_labels'] = label_lists['glabel']
    dataset['q_to_k_index'] = label_lists['k_q_index']
    dataset['k_qnum'] = label_lists['k_qnum']

    print(f'total q:')
    print(dataset['k_qnum'][-1])
    print(f'total k: {len(graph_k_list)}')

    total_aug_num = 0
    for idx in range(dataset['k_qnum'][-1]):
        k_idx = int(dataset['q_to_k_index'][idx])
        q_idx = int(idx - dataset['k_qnum'][k_idx])

        model_path = GRAPH_SUB_AUG_INPUT_PATH + 'aug_' + str(k_idx) + '.bin'
        graph_q_set = load_graphs(model_path)[0]

        # total_aug_num += len(model_path)
        # print(f'-{k_idx}/{idx}--{q_idx}/{len(graph_q_set[0])}-{len(graph_q_set[1])}')
        # print(graph_q_set[0])
    print(f'total_aug_num :{total_aug_num}')
