# -*- coding: utf-8 -*-
# @Author :ZhenZiyang

import pymongo
from bson import json_util as jsonb
from bson.objectid import ObjectId
import json
import time
import os
import re
import datetime
import pickle
import logging

GRAPH_OUTPUT_PATH = './graph.pkl'
NODE_OUTPUT_PATH = './nodes.pkl'

graph = {}
Nodes = []

class Sample:
    def __init__(self,num,name,label,family):#,label_code,family_code):
        self.num = num
        self.name = name
        self.label = label
        self.family = family
        self.key = '' # 新加的
        # self.label_code = label_code
        # self.family_code = family_code
        # num 编号, name file_hash, label 大类, family 小类,两个code 编号

class Node:
    def __init__(self,num,name,type_,sample,pid,key):
        self.num = num
        self.name = name
        self.type_ = type_
        self.sample = sample
        self.pid = pid
        self.key = key



with open(GRAPH_OUTPUT_PATH, 'rb') as fr:
    graph = pickle.load(fr)

with open(NODE_OUTPUT_PATH, 'rb') as f:
    Nodes = pickle.load(f)

num = 0
for key in graph:
    if key[0] == 's':
        num += 1

print(f'graph的sample中： 开头带s的个数 : {num}， 总个数： {len(graph)}')
process_size = 0

for node in Nodes:
    if node.type_ == 'process':
        process_size += 1
        # print(node.name)
print(f'process node: {process_size}')


# sample里怎么还有process id，而不是带s的key
def scatter_nodes():
    node_index_dict = {} # node_index_dict[node.num] = index 
    index_node_list = []
    # node_num to index
    sca_nodes = {'process':[], 'sign':[], 'network':[], 'memory':[], 'file':[], 'reg':[], 'api':[]}
    for node in Nodes:
        sca_nodes[node.type_].append(node.num)
 
    index = 0

    for node_type in sca_nodes:
        print(f'num of {node_type}: {len(sca_nodes[node_type])}')
        # print(f'{node_type} begin with index {index}')
        # print(f'{node_type} node sum {len(sca_nodes[node_type])}')
        if node_type == 'process':
            process_size = len(sca_nodes[node_type])

        for node_num in sca_nodes[node_type]:
            if node_num in node_index_dict:
                logging.warn(f'{node_type} type的重复出现了 {node_num}，{Nodes[node_num]}')
            node_index_dict[node_num] = index 
            index_node_list.append(node_num)
            index += 1
        # print(f'{node_type} end with index {index}')
    return node_index_dict, index_node_list


def gen_matrix():
    """ 这有一点需要注意： 
    sample和process是不完全重叠的，
    比如node中的process的节点个数比sample少。但不应该是这个，应该有链接就画上
    因此当访问sample的时候，并没有当作一个process记录下来，而只是记录子进程
    """
    matrix = []
    sample_num_index = {}
    
    api_matrix, api_index_map = get_api_matrix()
    api_argu_len = len(api_matrix[0])

    node_index_dict, index_node_list = scatter_nodes()
    print(f'{len(node_index_dict)} nodes in total')
    print(f'samples: {len(graph)}')
    for temp_node_num in index_node_list:
    # for samples in graph: 
        list = [0]*(len(node_index_dict)+api_argu_len) # 初始化长度全0，长度应该和node_index_dict长度一样
        if Nodes[temp_node_num].key == '':
            matrix.append(list)
            continue
        samples = Nodes[temp_node_num].key
        # print(f"find sample {Nodes[temp_node_num].key} in matrix~!")
        # if node对应的是sample
        for type in graph[samples]:
            node_set = graph[samples][type]
            for node_num in node_set:
                if node_index_dict[node_num] >= len(node_index_dict):
                    logging.warn(f'index {node_num} out of len of list: {node_index_dict[node_num]}')
                else:
                    list[node_index_dict[node_num]] = 1

                # 判断是否是api类型的 
                if type == 'api':
                    # 得到api name： Nodes[node_num].name
                    api_matrix_index = api_index_map[Nodes[node_num].name]
                    # 找到对应的array  api_matrix[api_matrix_index]
                    i = 0
                    for num in api_matrix[api_matrix_index]:
                        # 长度超长，报错
                        if len(node_index_dict)+i-1 > len(list): 
                            logging.warn(f'api len : {len(api_matrix[api_matrix_index])}, list len {len(list)}')
                        list[len(node_index_dict)+i-1] = num
                        i += 1

        sample_num_index[samples] = len(matrix)
        matrix.append(list)

    return matrix, api_argu_len


def gen_enhance_del_matrix():
    j = 1
    matrix, api_argu_len = gen_matrix()
    print()
    print(len(matrix))
    print(len(matrix[0]))
    # matrix对应的sample key的方式: graph[Nodes[index_node_list[i]].key]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])-api_argu_len):
            # 找到pa, pf这样的点，删除p与a, p与f的联系。只保留关键路径
            find_legal = False
            for k in range(len(matrix)):
                if matrix[k][j] == 1:
                    find_legal = True
                    break
            if find_legal == False:
                matrix[i][j] = 0
                matrix[j][i] = 0 # 是否增强有待考虑
    return matrix  


def gen_enhance_add_matrix():
    j = 1
    matrix, api_argu_len = gen_matrix()
    print()
    print(len(matrix))
    print(len(matrix[0]))
    print(process_size)
    # matrix对应的sample key的方式: graph[Nodes[index_node_list[i]].key]
    for i in range(process_size):
        print(f'enhance line {i}')
        for j in range(len(matrix[0])-api_argu_len):
            # 让pap, pfp这样的组合中的两个p建立联系
            for k in range(process_size):
                if matrix[k][j] == 1:
                    matrix[i][k] = 1
                    matrix[k][i] = 1 # 是否增强有待考虑
    return matrix  


def get_api_matrix():
    OUTPUT_PATH = "./mid_data/api_matrix1_3.pkl"
    with open(OUTPUT_PATH, 'rb') as fr:
        api_matrix = pickle.load(fr)
        api_index_map = pickle.load(fr)
    return api_matrix, api_index_map



gen_matrix()