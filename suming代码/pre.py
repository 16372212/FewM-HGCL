# -*- coding: utf-8 -*-

import pymongo
from bson import json_util as jsonb
import json
import time
import os
import re
f = open('../label/sample_result.txt','r')
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
class Sample:
    def __init__(self,num,name,label,family):#,label_code,family_code):
        self.num = num
        self.name = name
        self.label = label
        self.family = family
        # self.label_code = label_code
        # self.family_code = family_code
        # num 编号, name file_hash, label 大类, family 小类,两个code 编号

class Node:
    def __init__(self,num,name,type_,sample,pid):
        self.num = num
        self.name = name
        self.type_ = type_
        self.sample = sample
        self.pid = pid
        #  num 编号, name 唯一标识, type_ 种类, sample 归属样本, pid 进程号, 后两个没用
        ## type_: process, file, reg, memory, sign, network
        ## process: x['behavior']['processtree'] to do dfs
        ## file   : x['behavior']['generic'][(数组下标)]['summary'][api] , api in ['file_created','file_written']
        ## reg    : x['behavior']['generic'][(数组下标)]['summary'][api] , api in ['regkey_read','regkey_opened'] 提取 {} 部分
        ## memory : x['behavior']['generic'][(数组下标)]['summary']['dll_loaded']
        ## sign   : x['signatures'][(数组下标)]['name']
        ## network: x['network']['tcp']['dst'] | x['network']['udp']['dst']
        ## 

## 深搜遍历，用于连接样本、进程
## 逻辑： sample 和 子process连接, 子process 之间相互可能连接
def dfs(process,sample):
    process_name = process['process_name'].replace(' .','.')
    if process_name=='cmd.exe':
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
            pronode = Node(len(Nodes),process_name,'process','','')
            Nodes.append(pronode)
        else:
            # pronode 赋值为 原有已建立好的节点
            pronode = Nodes[process_map[process_name]]
        current = pronode

    if 'children' in process:
        # dfs遍历所有子节点
        for children in process['children']:
            childnode = dfs(children,sample)
            # 如果子节点不是sample样本本身，则建立新的连接
            if childnode:
                connect(current,childnode)
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
def connect(node1,node2):
    if node1.num == node2.num:
        return

    key = str(node1.num)
    if type(node1) == Sample:
        key = 's' + key

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
database_name = "cuckoo_nfs_db2"
collection_name = "analysis"
client = pymongo.MongoClient(host=ip, port=port,unicode_decode_error_handler='ignore')
dblist = client.list_database_names()
# collections = client[database_name][collection_name]

# static_collection = client['static_info_db']
# result_collection = client['labels']

# 用于存储所有的sample
sample_list = []

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

databases_name = ["cuckoo_nfs_db","cuckoo_nfs_db2"]#,"cuckoo_nfs_db3","cuckoo_nfs_db4","cuckoo_nfs_db5"]
for database_name in databases_name:
    collections = client[database_name][collection_name]
    file_collection = client[database_name]['report_id_to_file']
    for x in collections.find():
        # 进程list,包括样本
        file_hash = ''

        # 1.获取hash
        rows = file_collection.find(filter={'_id':str(x['_id'])})
        for row in rows:
            file_hash = row['file_hash']
        if file_hash not in labels:
            continue

        # 2.建立sample节点，并放入sample_list中
        sample = Sample(len(sample_list),file_hash,labels[file_hash]['label'],labels[file_hash]['family'])
        sample_list.append(sample)

        # 3.构图 - 进程
        if 'behavior' in x and 'processtree' in x['behavior']:
            processtree = x['behavior']['processtree']
            del(processtree[0])
            for process in processtree:
                dfs(process,sample)

        # 4.1 构图 - 文件
        file_name_list = ['file_created','file_written']
        if 'behavior' in x and 'generic' in x['behavior']:
            generics = x['behavior']['generic']
            for generic in generics:
                pro_name = generic['process_name'].replace(' .','.')
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
                                file = re.findall('\\\\([^\\\\]*)$',file)[0]
                            file = file.replace(' .','.')
                            if len(file)>15 or ' ' in file or '.' not in file or '.tmp' in file or '.dll' in file:
                                continue
                            if file not in file_list:
                                file_list.add(file)
                                file_map[file] = len(Nodes)
                                filenode = Node(len(Nodes),file,'file','',-1)
                                Nodes.append(filenode)
                            else:
                                filenode = Nodes[file_map[file]]

                            connect(process,filenode)        

        # 4.2 构图 - 注册表
        reg_name_list = ['regkey_read','regkey_opened']
        if 'behavior' in x and 'generic' in x['behavior']:
            generics = x['behavior']['generic']
            for generic in generics:
                pro_name = generic['process_name'].replace(' .','.')
                if pro_name=='lsass.exe':
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
                                reg = re.findall('{(.*?)}',reg)[0]
                            else:
                                continue
                            # if '\\\\' in reg or 'C:\\' in reg or '\\' in reg:
                            #     reg = re.findall('\\\\([^\\\\]*)$',reg)[0]
                            # if file_hash in reg or len(reg)>15:
                            #     continue
                            if reg not in reg_list:
                                reg_list.add(reg)
                                reg_map[reg] = len(Nodes)
                                regnode = Node(len(Nodes),reg,'reg','',-1)
                                Nodes.append(regnode)
                            else:
                                regnode = Nodes[reg_map[reg]]
                            connect(process,regnode)

        # 4.3 构图 - memory 内存加载程序
        if 'behavior' in x and 'generic' in x['behavior']:
            generics = x['behavior']['generic']
            for generic in generics:
                pro_name = generic['process_name'].replace(' .','.')
                if pro_name=='lsass.exe':
                    continue

                process = ''
                # 可能是样本或子进程
                if file_hash in pro_name:
                    process = sample
                elif pro_name in process_map:
                    process = Nodes[process_map[pro_name]]
                else:
                    continue

                if 'summary' in generic and 'dll_loaded' in generic['summary']:
                    for memory in generic['summary']['dll_loaded']:
                        if '\\\\' in memory or 'C:\\' in memory or '\\' in memory:
                            memory = re.findall('\\\\([^\\\\]*)$',memory)[0]
                        if file_hash in memory or len(memory)>15:
                            continue
                        if memory not in memory_list:
                            memory_list.add(memory)
                            memory_map[memory] = len(Nodes)
                            memorynode = Node(len(Nodes),memory,'memory','',-1)
                            Nodes.append(memorynode)
                        else:
                            memorynode = Nodes[memory_map[memory]]
                        connect(process,memorynode)

        # 4.4 构图 - 签名
        if 'signatures' in x:
            signatures = x['signatures']
            for signature in signatures:
                if 'name' in signature:
                    sign = signature['name']
                    if sign not in sign_list:
                        sign_list.add(sign)
                        sign_map[sign] = len(Nodes)
                        signnode = Node(len(Nodes),sign,'sign','',-1)
                        Nodes.append(signnode)
                    else:
                        signnode = Nodes[sign_map[sign]]
                    # 该特征直接与sample相连
                    connect(sample,signnode)

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
                    netnode = Node(len(Nodes),net,'network','',-1)
                    Nodes.append(netnode)
                else:
                    netnode = Nodes[network_map[net]]
                connect(sample,netnode)        
        
# print('graph:')
# print(graph)

f = open('pre1.txt','a+',encoding='utf-8')
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
    f.write(str(sample_list[i].num) + "," + str(sample_list[i].name) + "," + str(sample_list[i].label) + "," + str(sample_list[i].family) + "\n")
f.close()
print("Write Sample Done")
# for file in file_path

# n * m 维度, n 样本个数, m Node 节点个数 = process + file + reg + memory + sign + network 
matrix = []
