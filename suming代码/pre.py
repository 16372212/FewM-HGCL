# -*- coding: utf-8 -*-
# @Author :SuMing

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
import copy

GRAPH_OUTPUT_PATH = './graph.pkl'
NODE_OUTPUT_PATH = './nodes.pkl'
SAMPLE_LIST_OUTPUT_PATH = './sample_list.pkl'
SAMPLE_SAMPLE_NUM_TO_NODE_ID = './sample_num_to_node_id.pkl'

f = open('../label/sample_result.txt', 'r')
datas = f.read().split('\n')

labels = {}
for data in datas:
    if ',' in data:
        data = data.split(',')
        file_hash = data[0]
        type1 = data[1]
        type2 = data[2]
        labels[file_hash] = {}
        labels[file_hash]['label'] = type1
        labels[file_hash]['family'] = type2

f.close()
print('Loaded Labels')

with open('../mid_data/api_index_map.pkl', 'rb') as fr:
    api_index_matrix = pickle.load(fr)

MAX_API_NUM = 50


class Sample:
    def __init__(self, num, name, label, family):  # ,label_code,family_code):
        self.num = num
        self.name = name
        self.label = label
        self.family = family
        self.key = ''  # 新加的
        # self.label_code = label_code
        # self.family_code = family_code
        # num 编号, name file_hash, label 大类, family 小类,两个code 编号


class Node:
    def __init__(self, num, name, type_, sample, pid, key):
        self.num = num
        self.name = name
        self.type_ = type_
        self.sample = sample
        self.pid = pid
        self.key = key
        #  num 编号, name 唯一标识, type_ 种类, sample 归属样本, pid 进程号, 后两个没用
        ## type_: process, file, reg, memory, sign, network
        ## process: x['behavior']['processtree'] to do dfs
        ## file   : x['behavior']['generic'][(数组下标)]['summary'][api] , api in ['file_created','file_written']
        ## reg    : x['behavior']['generic'][(数组下标)]['summary'][api] , api in ['regkey_read','regkey_opened'] 提取 {} 部分
        ## memory : x['behavior']['generic'][(数组下标)]['summary']['dll_loaded']
        ## sign   : x['signatures'][(数组下标)]['name']
        ## network: x['network']['tcp']['dst'] | x['network']['udp']['dst']
        ##


def analyze_nodes():
    print('------------analyze nodes------------')
    print(f' {len(Nodes)} nodes in total')
    type_dict = {}

    l = len(Nodes)

    for i in range(l):
        if Nodes[i].type_ not in type_dict:
            type_dict[Nodes[i].type_] = 1
        else:
            type_dict[Nodes[i].type_] += 1

    for type_num in type_dict:
        print(f'{type_num} : {type_dict[type_num]}')


def ana_samples():
    """处理收集到的所有sample的个数"""
    print(f'actual sample lists: {len(sample_list)}')
    actual_familys = {}
    for samples in sample_list:
        if samples.family not in actual_familys:
            actual_familys[samples.family] = 1
        else:
            actual_familys[samples.family] += 1
    print(f'actual collected familys this time: {len(actual_familys)}')
    print(actual_familys)


def ana_total_labels():
    """这里总结的所有的数据里的family的个数。而不是实际上此次收集的数据集中family的个数"""
    print('------------analyze labels------------')
    print(f'total sample num: {len(labels)}')
    label_dict = {}
    family_dict = {}

    for file_hash in labels:
        if labels[file_hash]['label'] not in label_dict:
            label_dict[labels[file_hash]['label']] = 1
        else:
            label_dict[labels[file_hash]['label']] += 1

        if labels[file_hash]['family'] not in family_dict:
            family_dict[labels[file_hash]['family']] = []

        family_dict[labels[file_hash]['family']].append(1)  # [labels[file_hash]['label']] = 1

    print(f'total label category : {len(label_dict)}')
    print(label_dict)
    # for label in label_dict:
    #     print(f'label {label} num : {label_dict[label]}')

    print()
    print(f'total family category : {len(family_dict)}')

    # for fam in family_dict:
    #     print(f'family {fam} num : {len(family_dict[fam])}')

    print(f'total samples in sample_list: {len(sample_list)}')


## 深搜遍历，用于连接样本、进程
## 逻辑： sample 和 子process连接, 子process 之间相互可能连接
def dfs(process, sample):
    process_name = process['process_name'].replace(' .', '.')
    if process_name == 'cmd.exe':
        return None
    current = ''
    # 当前process 是样本本身
    if sample.name in process_name:
        current = sample
    # 当前process 是子进程
    else:
        if process_name not in process_list:
            process_list.append(process_name)
            process_map[process_name] = len(Nodes)
            pronode = Node(len(Nodes), process_name, 'process', '', 0, '')
            Nodes.append(pronode)
        else:
            # pronode 赋值为 原有已建立好的节点
            pronode = Nodes[process_map[process_name]]
        current = pronode
        connect(sample, pronode)  # 为啥这个要链接

    if 'children' in process:
        # dfs遍历所有子节点
        for children in process['children']:
            childnode = dfs(children, sample)
            # 如果子节点不是sample样本本身，则建立新的连接
            if childnode:
                connect(sample, childnode)
                # ******** 适当修改
                # connect(current,childnode)

    # 若当前节点是样本本身，则不返回内容，以免重复连接
    if current != sample:
        return current
    return None


## 深搜遍历，用于连接样本、进程
## 逻辑： sample 和 子process连接, 子process 之间相互可能连接
def old_dfs(process, sample):
    process_name = process['process_name'].replace(' .', '.')
    if process_name == 'cmd.exe':
        return None
    current = ''
    # 当前process 是样本本身
    if sample.name in process_name:
        current = sample
    # 当前process 是子进程
    else:
        if process_name not in process_list:
            process_list.append(process_name)
            process_map[process_name] = len(Nodes)
            pronode = Node(len(Nodes), process_name, 'process', '', '')
            Nodes.append(pronode)
        else:
            # pronode 赋值为 原有已建立好的节点
            pronode = Nodes[process_map[process_name]]
        current = pronode

    if 'children' in process:
        # dfs遍历所有子节点
        for children in process['children']:
            childnode = dfs(children, sample)
            # 如果子节点不是sample样本本身，则建立新的连接
            if childnode:
                connect(current, childnode)
    # 若当前节点是样本本身，则不返回内容，以免重复连接
    if current != sample:
        return current
    return None
    # if sample.name in process_name:
    #     if process_name not in process_list:
    #         process_list.append(process_name)
    #         process_num[process_name] = sample.num
    #         sample.pid = process['pid']
    #     current = sample
    # else:
    #     if process_name not in process_list:
    #         process_list.append(process_name)
    #         process_num[process_name] = len(Nodes)
    #         pronode = Node(len(Nodes),process_name,'process','',process['pid'])
    #         Nodes.append(pronode)
    #     else:
    #         pronode = Nodes[process_num[process_name]]
    #     current = pronode

    # if 'children' in process:
    #     for children in process['children']:
    #         # build(process)
    #         childnode = dfs(children,sample)
    #         if childnode:
    #             connect(current,childnode)
    # return current


## 连接操作
def connect(node1, node2):
    if node1.num == node2.num and type(node1) != Sample:
        return

    # zz修改：为方便绘图，如果是两个process节点，就让两个process节点交换顺序，从而让reg或file指向process
    if type(node1) != Sample:
        temp = copy.deepcopy(node1)
        node1 = copy.deepcopy(node2)
        node2 = temp

    key = str(node1.num)
    if type(node1) == Sample:
        key = 's' + key
        node1.key = key  # 新设置的key

    if key not in graph:
        graph[key] = {}
    if node2.type_ not in graph[key]:
        graph[key][node2.type_] = set()
    if node2.num not in graph[key][node2.type_]:
        graph[key][node2.type_].add(node2.num)
    # if node1.num not in graph:
    #     graph[node1.num] = {}
    # if node2.type_ not in graph[node1.num]:
    #     graph[node1.num][node2.type_] = set()
    # if node2.num not in graph[node1.num][node2.type_]:
    #     graph[node1.num][node2.type_].add(node2.num)
    # # 双向
    # if node2.num not in graph:
    #     graph[node2.num] = {}
    # if node1.type_ not in graph[node2.num]:
    #     graph[node2.num][node1.type_] = set()
    # if node1.num not in graph[node2.num][node1.type_]:
    #     graph[node2.num][node1.type_].add(node1.num)


graph = {}
Nodes = []
ip = "192.168.105.224"
port = 27017
# database_name = "cuckoo_nfs_db2"
collection_name = "analysis"
client = pymongo.MongoClient(host=ip, port=port, unicode_decode_error_handler='ignore')
dblist = client.list_database_names()
# collections = client[database_name][collection_name]

# static_collection = client['static_info_db']
# result_collection = client['labels']

# 用于存储所有的sample
sample_list = []

# 用来存储所有的sample_num 与 node_id的对应关系
sample_num_to_node_id = {}

# list用于存储所有的Node实体名称(唯一标识),用于快速检测是否存在,是否需要重新创建新的Node;
# map对应 {实体 Node: 编号 id},用于快速检索在graph中的位置
process_list = []
process_map = {}
file_list = set()
file_map = {}
reg_list = set()
reg_map = {}
memory_list = set()
memory_map = {}
sign_list = set()
sign_map = {}
network_list = set()
network_map = {}

sample_label = {}  # 用来存储所有的sample对应的list
MAX_FAMILY_NUM_THREAD = 20

databases_name = ["cuckoo_nfs_db", "cuckoo_nfs_db2", "cuckoo_nfs_db3", "cuckoo_nfs_db4"]  # ,"cuckoo_nfs_db5"]
dbcalls_dict = {'cuckoo_nfs_db': 'from_nfs_db1', 'cuckoo_nfs_db2': 'from_nfs_db2', 'cuckoo_nfs_db3': 'from_nfs_db3',
                'cuckoo_nfs_db4': 'from_nfs_db4'}
for database_name in databases_name:
    collections = client[database_name][collection_name]
    file_collection = client[database_name]['report_id_to_file']  # 获取所有的file, hash映射，可以看作一个dict
    call_collection = client['db_calls'][dbcalls_dict[database_name]]
    cursor = collections.find(no_cursor_timeout=True)
    for x in cursor:
        # 进程list,包括样本
        file_hash = ''

        # 1.获取hash
        rows = file_collection.find(filter={'_id': str(x['_id'])})
        for row in rows:
            file_hash = row['file_hash']
        if file_hash not in labels:
            continue

        # ********************
        # zz修改：graph与node个数的限制。让每个family对应的sample的个数都均衡起来。这样就让个数尽量接近（20*112）
        # if 这个family类型的树木超过了20，就可以不加入了
        if labels[file_hash]['family'] in sample_label:
            if sample_label[labels[file_hash]['family']] < MAX_FAMILY_NUM_THREAD:
                sample_label[labels[file_hash]['family']] += 1
            else:
                continue
        else:
            sample_label[labels[file_hash]['family']] = 1

        # 2.建立sample节点，并放入sample_list中
        sample = Sample(len(sample_list), file_hash, labels[file_hash]['label'], labels[file_hash]['family'])
        sample_list.append(sample)

        # 3.构图 - 进程
        if 'behavior' in x and 'processtree' in x['behavior']:
            processtree = x['behavior']['processtree']
            del (processtree[0])
            for process in processtree:
                dfs(process, sample)

        # 4.1 构图 - 文件
        file_name_list = ['file_created', 'file_written']
        if 'behavior' in x and 'generic' in x['behavior']:
            generics = x['behavior']['generic']
            for generic in generics:
                pro_name = generic['process_name'].replace(' .', '.')
                if pro_name == 'lsass.exe':
                    continue

                process = ''
                # 可能是样本或子进程
                if file_hash in pro_name:
                    process = sample
                elif pro_name in process_map:
                    process = Nodes[process_map[pro_name]]
                else:
                    continue

                if 'summary' in generic:
                    for file_type in file_name_list:
                        if file_type not in generic['summary']:
                            continue
                        for file in generic['summary'][file_type]:
                            if file_hash in file:
                                continue

                            if '\\\\' in file or 'C:\\' in file or 'c:\\' in file:
                                file = re.findall('\\\\([^\\\\]*)$', file)[0]
                            file = file.replace(' .', '.')
                            if len(file) > 15 or ' ' in file or '.' not in file or '.tmp' in file or '.dll' in file:
                                continue
                            if file not in file_list:
                                file_list.add(file)
                                file_map[file] = len(Nodes)
                                filenode = Node(len(Nodes), file, 'file', '', -1, '')
                                Nodes.append(filenode)
                            else:
                                filenode = Nodes[file_map[file]]

                            connect(process, filenode)


                            # 4.2 构图 - 注册表
        reg_name_list = ['regkey_read', 'regkey_opened']
        if 'behavior' in x and 'generic' in x['behavior']:
            generics = x['behavior']['generic']
            for generic in generics:
                pro_name = generic['process_name'].replace(' .', '.')
                if pro_name == 'lsass.exe':
                    continue

                process = ''
                # 可能是样本或子进程
                if file_hash in pro_name:
                    process = sample
                elif pro_name in process_map:
                    process = Nodes[process_map[pro_name]]
                else:
                    continue

                if 'summary' in generic:
                    for reg_type in reg_name_list:
                        if reg_type not in generic['summary']:
                            continue
                        for reg in generic['summary'][reg_type]:
                            if file_hash in reg:
                                continue
                            if '{' in reg and '}' in reg:
                                reg = re.findall('{(.*?)}', reg)[0]
                            else:
                                continue
                            # if '\\\\' in reg or 'C:\\' in reg or '\\' in reg:
                            #     reg = re.findall('\\\\([^\\\\]*)$',reg)[0]
                            # if file_hash in reg or len(reg)>15:
                            #     continue
                            if reg not in reg_list:
                                reg_list.add(reg)
                                reg_map[reg] = len(Nodes)
                                regnode = Node(len(Nodes),reg,'reg','',-1,'')
                                Nodes.append(regnode)
                            else:
                                regnode = Nodes[reg_map[reg]]
                            connect(process, regnode)

        # 4.3 构图 - memory 内存加载程序
        # if 'behavior' in x and 'generic' in x['behavior']:
        #     generics = x['behavior']['generic']
        #     for generic in generics:
        #         pro_name = generic['process_name'].replace(' .','.')
        #         if pro_name=='lsass.exe':
        #             continue
        #
        #         process = ''
        #         # 可能是样本或子进程
        #         if file_hash in pro_name:
        #             process = sample
        #         elif pro_name in process_map:
        #             process = Nodes[process_map[pro_name]]
        #         else:
        #             continue
        #
        #         if 'summary' in generic and 'dll_loaded' in generic['summary']:
        #             for memory in generic['summary']['dll_loaded']:
        #                 if '\\\\' in memory or 'C:\\' in memory or '\\' in memory:
        #                     memory = re.findall('\\\\([^\\\\]*)$',memory)[0]
        #                 if file_hash in memory or len(memory)>15:
        #                     continue
        #                 if memory not in memory_list:
        #                     memory_list.add(memory)
        #                     memory_map[memory] = len(Nodes)
        #                     memorynode = Node(len(Nodes),memory,'memory','',-1,'')
        #                     Nodes.append(memorynode)
        #                 else:
        #                     memorynode = Nodes[memory_map[memory]]
        #                 connect(process,memorynode)

        # 4.4 构图 - 签名
        # if 'signatures' in x:
        #     signatures = x['signatures']
        #     for signature in signatures:
        #         if 'name' in signature:
        #             sign = signature['name']
        #             if sign not in sign_list:
        #                 sign_list.add(sign)
        #                 sign_map[sign] = len(Nodes)
        #                 signnode = Node(len(Nodes), sign, 'sign', '', -1, '')
        #                 Nodes.append(signnode)
        #             else:
        #                 signnode = Nodes[sign_map[sign]]
        #             # 该特征直接与sample相连
        #             connect(sample, signnode)

        # 4.5 构图 - 网络
        network = []
        if 'network' in x:
            if 'tcp' in x['network']:
                network_tcp = x['network']['tcp']
                for net in network_tcp:
                    item = str(net['dst'])
                    if item not in network:
                        network.append(item)
            if 'udp' in x['network']:
                network_udp = x['network']['udp']
                for net in network_udp:
                    item = str(net['dst'])
                    if item not in network:
                        network.append(item)

            for net in network:
                if net not in network_list:
                    network_list.add(net)
                    network_map[net] = len(Nodes)
                    netnode = Node(len(Nodes), net, 'network', '', -1, '')
                    Nodes.append(netnode)
                else:
                    netnode = Nodes[network_map[net]]
                connect(sample, netnode)

                # 4.6构图 - 获取api节点，直接链接sample
        call_rows = call_collection.find(filter={'_id': x['_id']})
        calls = {}

        for call_row in call_rows:
            calls = call_row['calls']

        api_num = 0
        for call in calls:
            # 先判断这个api是否在里面
            if call not in api_index_matrix:
                continue
            apinode = Node(len(Nodes), call, 'api', '', 0, '')  # 将api对应的转到id， 这个id不太对
            connect(sample, apinode)
            Nodes.append(apinode)
            api_num += 1
            if api_num > MAX_API_NUM:
                # print(f'api num more than {MAX_API_NUM}')
                break

        # *****zz修改：******
        # 将sample也加到node里，因为sample也算是一种process，这样为了构造matrix方便。
        # 得到sample.num -> len(Nodes)
        if sample.key != '':
            sample_node = Node(len(Nodes), file_hash, 'process', '', -1, sample.key)
            sample_num_to_node_id[sample.num] = len(Nodes)
            Nodes.append(sample_node)
        else:
            sample_node = Node(len(Nodes), file_hash, 'process', '', -1, '')
            sample_num_to_node_id[sample.num] = len(Nodes)
            Nodes.append(sample_node)
            logging.warning('出现了sample没有key的情况')

    cursor.close()
    print(f"in database {database_name}")

analyze_nodes()
ana_samples()
# print('graph:')
# print(graph)


# save graph to pkl
with open(GRAPH_OUTPUT_PATH, 'wb') as fr:
    pickle.dump(graph, fr)

with open(NODE_OUTPUT_PATH, 'wb') as fr:
    pickle.dump(Nodes, fr)

with open(SAMPLE_LIST_OUTPUT_PATH, 'wb') as fr:
    pickle.dump(sample_list, fr)

with open(SAMPLE_SAMPLE_NUM_TO_NODE_ID, 'wb') as fr:
    pickle.dump(sample_num_to_node_id, fr)

today = datetime.date.today()
f = open(f'pre{today}.txt', 'a+', encoding='utf-8')
f.write(str(graph))
f.write('\n')
print("Write Graph Done")
l = len(Nodes)

f.write('\nNodes:\n')

for i in range(l):
    f.write(str(Nodes[i].num) + "," + str(Nodes[i].type_) + "," + str(Nodes[i].name) + "\n")
print("Write Node Done")

f.write('\nSample:\n')
l = len(sample_list)
for i in range(l):
    f.write(str(sample_list[i].num) + "," + str(sample_list[i].name) + "," + str(sample_list[i].label) + "," + str(
        sample_list[i].family) + "\n")
f.close()
print("Write Sample Done")
# for file in file_path

# n * m 维度, n 样本个数, m Node 节点个数 = process + file + reg + memory + sign + network
matrix = []

print('sample num to node id map: ')
print(sample_num_to_node_id)
