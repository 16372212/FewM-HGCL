# -*- coding: utf-8 -*-

import pymongo
import json
# from bson.objectid import ObjectId
# import pandas as pd
import re
import time
import numpy as np

class Node:
    def __init__(self,num,name,type_,sample,pid):
        self.num = num
        self.name = name
        self.type_ = type_
        self.sample = sample
        self.pid = pid
        ## type_: sample, process, file, reg, network, sign

ip = "192.168.105.224"
port = 27017
database_name = "cuckoo_nfs_db4"
collection_name = "analysis"
client = pymongo.MongoClient(host=ip, port=port,unicode_decode_error_handler='ignore')
dblist = client.list_database_names()
if database_name in dblist:
    database = client[database_name]
    collections = database[collection_name]

# label_collection = client['labels']['reportid_to_single_label']
count = 2000
n_neighbours = 4
counts = 0
file_collection = client[database_name]['report_id_to_file']
static_collection = client['static_info_db']
result_collection = client['labels']

## 深搜遍历，用于连接样本、进程
def dfs(process,sample):
    # process_list = []
    # process_num = {}
    process_name = process['process_name'].replace(' .','.')
    if process_name=='cmd.exe':
        return None
    current = ''
    if sample.name in process_name:
        if process_name not in process_list:
            process_list.append(process_name)
            process_num[process_name] = sample.num
            sample.pid = process['pid']
        current = sample
    else:
        if process_name not in process_list:
            process_list.append(process_name)
            process_num[process_name] = len(Nodes)
            pronode = Node(len(Nodes),process_name,'process','',process['pid'])
            Nodes.append(pronode)
        else:
            pronode = Nodes[process_num[process_name]]
        current = pronode

    if 'children' in process:
        for children in process['children']:
            # build(process)
            childnode = dfs(children,sample)
            if childnode:
                connect(current,childnode)
    return current

## 连接操作
def connect(node1,node2):
    if node1.num == node2.num:
        return
    if node1.num not in graph:
        graph[node1.num] = {}
    if node2.type_ not in graph[node1.num]:
        graph[node1.num][node2.type_] = set()
    if node2.num not in graph[node1.num][node2.type_]:
        graph[node1.num][node2.type_].add(node2.num)
    # 双向
    if node2.num not in graph:
        graph[node2.num] = {}
    if node1.type_ not in graph[node2.num]:
        graph[node2.num][node1.type_] = set()
    if node1.num not in graph[node2.num][node1.type_]:
        graph[node2.num][node1.type_].add(node1.num)
    # if node2.num not in graph:
    #     graph[node2.num] = {}
    # if node1.type_ not in graph[node2.num]:
    #     graph[node2.num][node1.type_] = []
    # if node1.num not in graph[node2.num][node1.type_]:
    #     graph[node2.num][node1.type_].append(node1.num)


# 节点集合 Nodes
Nodes = []
# 大图 graph
graph = {}
# xxx_list存放名称,用于检测是否存在,故为列表格式;xxx_num存放{实体:编号id},用于快速检索在Nodes中位置
# 签名list
opcode_list = set()
opcode_num = {}
# 网络list
network_list = set()
network_num = {}
# 注册表list
reg_list = set()
reg_num = {}
# 文件list
file_list = set()
file_num = {}
# 样本list
sample_list = []
sample_num = []

trojan = 0
worm = 0
virus = 0
general = 0
time_start = time.time()
for x in collections.find():
    # 进程list,包括样本
    process_list = []
    process_num = {}
    file_hash = ''

    # 1.获取hash
    rows = file_collection.find(filter={'_id':str(x['_id'])})
    # print(type(rows))
    # print(dict(rows))
    # file_hash = dict(rows)['file_hash']
    for row in rows:
        # print('file_hash: ',row['file_hash'])
        file_hash = row['file_hash']
    if file_hash =='':
        continue
    

    #! 2.建立样本节点
    # build(sample)
    results = result_collection['reportid_to_single_label'].find(filter={'file_hash':file_hash})
    result = ''
    for result_ in results:
        result = result_['label']
    if result == 'trojan':
        trojan = trojan + 1
    elif result == 'virus':
        virus = virus + 1
    elif result == 'worm':
        worm = worm + 1
    else:
        general = general + 1
    sample = Node(len(Nodes),file_hash,'sample',result,-1)
    # time.sleep(0.15)
    # num,name,type_,sample,pid
    sample_list.append(sample)
    sample_num.append(len(Nodes))
    Nodes.append(sample)

    # 3.特征：获取签名
    sign_ids = static_collection['filename_mongo_reportid_dict'].find(filter={'_id':str(x['_id'])})
    for sign_id in sign_ids:
        opcode_id = sign_id['opcode_mongoid']
        opcodes_collection = static_collection['malware_op_list'].find(filter={'_id':opcode_id})
        for item in opcodes_collection:
            opcodes = item['opcodes']
            for opcode in opcodes:
                if opcode not in opcode_list:
                    opcode_list.add(opcode)
                    opcode_num[opcode] = len(Nodes)
                    opnode = Node(len(Nodes),opcode,'sign','',-1)
                    Nodes.append(opnode)
                    # num,name,type_,sample,pid
                else:
                    opnode = Nodes[opcode_num[opcode]]
                    # 建立opcode
                    # build(opcode)
                connect(sample,opnode)

    # 3.特征：获取网络
    network = []
    if 'network' in x:
        if 'tcp' in x['network']:
            network_tcp = x['network']['tcp']
            for net in network_tcp:
                item = str(net['dst'])
                # item = str(net['dst']) + ':' + str(net['dport'])
                if item not in network:
                    network.append(item)
        if 'udp' in x['network']:
            network_udp = x['network']['udp']
            for net in network_udp:
                item = str(net['dst'])
                # item = str(net['dst']) + ':' + str(net['dport'])
                if item not in network:
                    network.append(item)

    for net in network:
        if net not in network_list:
            network_list.add(net)
            network_num[net] = len(Nodes)
            netnode = Node(len(Nodes),net,'network','',-1)
            Nodes.append(netnode)
            # num,name,type_,sample,pid
        else:
            netnode = Nodes[network_num[net]]
            # 建立network
            # build(network)
        connect(sample,netnode)
    # print(opcode_num)
    # print(network_num)
    # print(graph)
    # print('----')
    # print(sample_num)
    # print('----')
    # build(network)
    # connect(sample,network)

    # 4.特征：获取进程
    # process = []
    if 'behavior' in x and 'processtree' in x['behavior']:
        processtree = x['behavior']['processtree']
        del(processtree[0])
        for process in processtree:
            dfs(process,sample)

    # # 4.特征：获取注册表
    if 'behavior' in x and 'generic' in x['behavior']:
        generics = x['behavior']['generic']
        for generic in generics:
            pro_name = generic['process_name'].replace(' .','.')
            if pro_name=='lsass.exe':
                continue
            if pro_name not in process_num:
                continue
            process = Nodes[process_num[pro_name]]
            if 'summary' in generic and 'dll_loaded' in generic['summary']:
                for reg in generic['summary']['dll_loaded']:
                    # print(reg)
                    if '\\\\' in reg or 'C:\\' in reg or '\\' in reg:
                        reg = re.findall('\\\\([^\\\\]*)$',reg)[0]
                    if file_hash in reg or len(reg)>20:
                        continue
                    if reg not in reg_list:
                        reg_list.add(reg)
                        reg_num[reg] = len(Nodes)
                        regnode = Node(len(Nodes),reg,'reg','',-1)
                        Nodes.append(regnode)
                        # num,name,type_,sample,pid
                    else:
                        regnode = Nodes[reg_num[reg]]
                        # 建立network
                        # build(network)
                        #
                    connect(process,regnode)
    # print(reg_list)
    # # print(reg_num)
    # 4.特征：获取文件
    # file = [] #'file_failed','file_exists'
    file_name_list = ['file_deleted','file_created','file_written','file_opened','file_read']
    # file = re.findall('\\\\([^\\\\]*)$',string)[0]

    if 'behavior' in x and 'generic' in x['behavior']:
        generics = x['behavior']['generic']
        for generic in generics:
            pro_name = generic['process_name'].replace(' .','.')
            if pro_name=='lsass.exe':
                continue
            if pro_name not in process_num:
                continue
            process = Nodes[process_num[pro_name]]
            if 'summary' in generic:
                for file_type in file_name_list:
                    if file_type not in generic['summary']:
                        continue
                    for file in generic['summary'][file_type]:
                        if file_hash in file:
                            continue
                        # print(file)
                        if '\\\\' in file or 'C:\\' in file or 'c:\\' in file:
                            file = re.findall('\\\\([^\\\\]*)$',file)[0]
                        file = file.replace(' .','.')
                        if len(file)>20 or ' ' in file or '.' not in file or '.tmp' in file:
                            continue
                        if file not in file_list:
                            file_list.add(file)
                            file_num[file] = len(Nodes)
                            filenode = Node(len(Nodes),file,'file','',-1)
                            Nodes.append(filenode)
                            # num,name,type_,sample,pid
                        else:
                            filenode = Nodes[file_num[file]]
                            # 建立network
                            # build(network)
                            #
                        connect(process,filenode)
    # print(process_list)
    # print(process_num)
    # print('file_list:')
    # print(file_list)
    # print(graph)

    counts = counts + 1
    if counts%count==0:
        print(counts)
        print(len(Nodes))
        time_end = time.time()
        print(time_end-time_start)
        break


# 5.构图

time.sleep(1)
l = len(Nodes)



# import networkx as nx
# import numpy
# Graph = nx.Graph()

# for node1 in graph:
#     for relation in graph[node1]:
#         for node2 in graph[node1][relation]:
#             Graph.add_edge(node1,node2)
# print('Graph.nodes.num:')
# print(len(Graph.nodes()))
# for i in Graph:
#     if len(Graph.nodes())<2:
#         Graph.remove_node(i)
# print('After Remove')
# print('Graph.nodes.num:')
# print(len(Graph.nodes()))



s = len(sample_list)
import numpy
matrix = np.zeros((l,l))

def set_value(matrix,source,target):
    matrix[source][target] = 1


# def do(matrix,graph,source):
#     graph_tmp = graph
#     nodes_list = graph_tmp.nodes()
#     # for node in nodes_list:
#     #     if Nodes[node].type_=='Sample' and source!=node:
#     #         graph_tmp.remove_node(node)
#     neighours_set = set()
#     current_set = set([source])
#     next_set = set()
#     for tmp in range(n_neighbours):
#         for i in current_set:
#             next_set = next_set | set(graph_tmp[i].keys())
#         neighours_set = neighours_set | next_set
#         current_set = next_set - current_set
#     for target in neighours_set:
#         matrix[source][target] = 1.00000000000000000000000000
#     next_set = set()
#     for i in current_set:
#         next_set = next_set | set(graph_tmp[i].keys())
#     for target in next_set:
#         matrix[source][target] = 0.10000000000000000000000000
#     print(sum(matrix[source]))

def do(matrix,graph,source):
    graph_tmp = graph
    nodes_list = graph_tmp.keys()
    # for node in nodes_list:
    #     if Nodes[node].type_=='Sample' and source!=node:
    #         graph_tmp.remove_node(node)
    neighours_set = set()
    next_set = set()
    out_set = set()
    visited = [i for i in range(l)]
    # if n_neighbours==6:

    if n_neighbours==4:
        for relation in graph[source]:
            for i in graph[source][relation]:
                next_set.add(i)
        # print(next_set)
        # print(type(next_set))
        for target1 in next_set:
            if target1 in neighours_set:
                continue
            ## visit 1
            if visited[target1]:
                continue
            visited[target1] = 1

            if Nodes[target1].type_=='process': 
            ## P - P
                neighours_set.add(target1)
                if 'process' in graph[target1] and len(graph[target1]['process'])>1: 
                    ## P - P - P
                    for target2 in graph[target1]['process']:
                        ## visit 2
                        if visited[target2]:
                            continue
                        visited[target2] = 1
                        neighours_set.add(target2)

                        if 'process' in graph[target2] and len(graph[target2]['process'])>1:
                            ## P - P - P - P
                            for target3 in graph[target2]['process']:
                                ## visit 3
                                if visited[target3]:
                                    continue
                                visited[target3] = 1
                                neighours_set.add(target3)
                                if 'process' in graph[target3] and len(graph[target3]['process'])>1:
                                    ## P - P - P - P - P
                                    for target4 in graph[target3]['process']:
                                        ## visit 4
                                        if visited[target4]:
                                            continue
                                        visited[target4] = 1
                                        neighours_set.add(target4)
                                        

                        for relation in graph[target2]:
                            ## P - P - P - F
                            if relation == 'process':
                                continue
                            for target3 in graph[target2][relation]:
                                ## visit 3
                                if visited[target3]:
                                    continue
                                visited[target3] = 1

                                if 'process' in graph[target3] and len(graph[target3]['process'])>1:
                                    ##  P - P - P - F - P
                                    for target4 in graph[target3]['process']:
                                        neighours_set.add(target3)
                                        neighours_set.add(target4)
                                        ## visit 4
                                        if visited[target4]:
                                            continue
                                        visited[target4] = 1

                                        
                else:
                    ## P - P - F 
                    for relation in graph[target1]:
                        if relation == 'process':
                            continue
                        for target2 in graph[target1][relation]:
                            ## visit 2
                            if visited[target2]:
                                continue
                            visited[target2] = 1
                
                            if 'process' in graph[target2] and len(graph[target2]['process'])>1:
                                ## P - P - F - P
                                for target3 in graph[target2]['process']:
                                    neighours_set.add(target2)
                                    neighours_set.add(target3)
                                    ## visit 3
                                    if visited[target3]:
                                        continue
                                    visited[target3] = 1
                
                                    if 'process' in graph[target3]:
                                        ## P - P - F - P - P
                                        for target4 in graph[target3]['process']:
                                            neighours_set.add(target4)
                                            
                                            ## visit 4
                                            if visited[target4]:
                                                continue
                                            visited[target4] = 1
            else: 
            ## P - F
                if 'process' in graph[target1] and len(graph[target1]['process'])>1:
                    ## P - F - P
                    for target2 in graph[target1]['process']:
                        neighours_set.add(target1)
                        neighours_set.add(target2)
                        ## visit 2
                        if visited[target2]:
                            continue
                        visited[target2] = 1

                        for relation in graph[target2]:
                            if relation == 'process':
                            ## P - F - P - P
                                for target3 in graph[target2]['process']:
                                    neighours_set.add(target3)
                                    ## visit 3
                                    if visited[target3]:
                                        continue
                                    visited[target3] = 1

                                    ## P - F - P - P - P
                                    if 'process' in graph[target3]:
                                        for target4 in graph[target3]['process']:
                                            neighours_set.add(target4)

                                            ## visit 4
                                            if visited[target4]:
                                                continue
                                            visited[target4] = 1
                            else:
                            ## P - F - P - F
                                for target3 in graph[target2][relation]:
                                    ## visit 3
                                    if visited[target3]:
                                        continue
                                    visited[target3] = 1

                                    ## P - F - P - F - P
                                    if 'process' in graph[target3] and len(graph[target3]['process'])>1:
                                        for target4 in graph[target3]['process']:
                                            neighours_set.add(target4)

                                            ## visit 4
                                            if visited[target4]:
                                                continue
                                            visited[target4] = 1

    elif n_neighbours==2:
        for relation in graph[source]:
            for i in graph[source][relation]:
                next_set.add(i)
        for target1 in next_set:
            if target1 in neighours_set:
                continue
            ## P - P
            if Nodes[target1].type_=='process':
                neighours_set.add(target1)
                ## P - P - P
                if 'process' in graph[target1] and len(graph[target1]['process'])>1:
                    for target2 in graph[target1]['process']:
                        neighours_set.add(target2)
                        out_set.add(target2)
            ## P - F
            else:
                ## P - F - P
                if 'process' in graph[target1] and len(graph[target1]['process'])>1:
                    for target2 in graph[target1]['process']:
                        neighours_set.add(target1)
                        neighours_set.add(target2)
                        out_set.add(target2)
    for target in neighours_set:
        matrix[source][target] = 1.0000000000000000
    for out in neighours_set:
        if out in graph:
            for relation in graph[out]:
                for target in graph[out][relation]:
                    if matrix[source][target] < 1:
                        matrix[source][target] = 0.1000000000000000
    # for tmp in range(n_neighbours):
    #     for i in current_set:
    #         next_set = next_set | set(graph_tmp[i].keys())
    #     for target in next_set:
    #         if target in neighours_set:
    #             continue
    #         if Nodes[target]=='process':
    #             neighours_set.add(target)
    #         else:
    #             if len(graph[target]['process'])
    #     neighours_set = neighours_set | next_set
    #     current_set = next_set - current_set
    # for target in neighours_set:
    #     matrix[source][target] = 1.00000000000000000000000000
    # next_set = set()
    # for i in current_set:
    #     next_set = next_set | set(graph_tmp[i].keys())
    # for target in next_set:
    #     matrix[source][target] = 0.10000000000000000000000000
    # print(sum(matrix[source]))




# 6.处理小图
print(len(opcode_list))
print('network_list:')
print(len(network_list))
print('file_list:')
print(len(file_list))
print('reg_list:')
print(len(reg_list))
# print('file_list\n\n\n\n\n\n\n\n')
# print(file_list)
# print('\n\n\n\n\n\n\n\n')
# print('opcode_list\n\n\n\n\n\n\n\n')
# print(opcode_list)
# print('\n\n\n\n\n\n\n\n')
# print('network_list\n\n\n\n\n\n\n\n')
# print(network_list)
# print('\n\n\n\n\n\n\n\n')
# print('reg_list\n\n\n\n\n\n\n\n')
# print(reg_list)
# print('\n\n\n\n\n\n\n\n')
# print('opcode_list:')


train_sample = numpy.zeros((s,l))
train_list = []
result_list = []
print('len_graph:')
print(l)
print('matrix.size:')
print(matrix.shape)
print('train_sample.size:')
print(train_sample.shape)

print('trojan:',trojan)
print('virus:',virus)
print('worm:',worm)
print('general:',general)
# trojan_sample = numpy.zeros((trojan,))

samples = trojan + virus + worm + general
if count==50:
    # cuckoo_db
    trojan_train = 22
    virus_train = 8
    worm_train = 8
    general_train = 2
    
    trojan_val = 4
    virus_val = 2
    worm_val = 2
    general_val = 0

elif count==500:
    # cuckoo_db
    trojan_train = 250
    virus_train = 78
    worm_train = 36
    general_train = 36

    trojan_val = 40
    virus_val = 28
    worm_val = 6
    general_val = 6

elif count==1000:
    trojan_train = 250
    virus_train = 78
    worm_train = 36
    general_train = 36

    trojan_val = 40
    virus_val = 28
    worm_val = 6
    general_val = 6

elif count==2000:
    # cuckoo_db4
    trojan_train = 970
    virus_train = 356
    worm_train = 168
    general_train = 106

    trojan_val = 200
    virus_val = 122
    worm_val = 42
    general_val = 36

elif count==5000:
    # cuckoo_db4
    trojan_train = 2484
    virus_train = 840
    worm_train = 396
    general_train = 280

    trojan_val = 500
    virus_val = 310
    worm_val = 120
    general_val = 70

trojan_test = trojan - trojan_train
virus_test = virus - virus_train
worm_test = worm - worm_train
general_test = general - general_train

trojan_train = trojan_train - trojan_val
virus_train = virus_train - virus_val
worm_train = worm_train - worm_val
general_train = general_train - general_val


trojan1 = numpy.zeros((trojan_train,l))
trojan2 = numpy.zeros((trojan_val,l))
trojan3 = numpy.zeros((trojan_test,l))
virus1 = numpy.zeros((virus_train,l))
virus2 = numpy.zeros((virus_val,l))
virus3 = numpy.zeros((virus_test,l))
worm1 = numpy.zeros((worm_train,l))
worm2 = numpy.zeros((worm_val,l))
worm3 = numpy.zeros((worm_test,l))
general1 = numpy.zeros((general_train,l))
general2 = numpy.zeros((general_val,l))
general3 = numpy.zeros((general_test,l))

trojan_index = 0
virus_index = 0
worm_index = 0
general_index = 0
'''
sample_index = 0
for source in graph:
    if Nodes[source].type_ == 'sample':
        do(matrix,Graph,source)
        if sum(matrix[source]):
            print(sum(matrix[source]))
        train_sample[sample_index] = matrix[source]
        sample_index = sample_index + 1
        result = Nodes[source].sample
        result_list.append(result)
 # = np.r_[a,b]

# np.save('samples.npy',train_sample)
with open('a.txt','w+') as f:
    for result in result_list:
        f.write(result+'\n')
f.close()
'''
import multiprocessing
#from itertools import product
#from pandarallel import pandarallel
#pandarallel.initialize()
start_time = time.time()
processors = 4
nums = 0
#def f(graph):
for source in graph:
    if Nodes[source].type_ == 'sample':
        #pool = multiprocessing.Pool(processes=processors)
        do(matrix,graph,source)
        #pool.map(do,product(matrix,Graph,source))
        #if sum(matrix[source]):
        #    print(sum(matrix[source]))
        result = Nodes[source].sample
        end_time = time.time()
        nums = nums + 1
        if nums%100==0:
            print("Sample",nums)
            # print(general_index)
            print(end_time - start_time)
        if result=='trojan':
            if trojan_index < trojan_train:
                trojan1[trojan_index] = matrix[source]
            elif trojan_index < trojan_train + trojan_val:
                trojan2[trojan_index - trojan_train] = matrix[source]
            else:
                trojan3[trojan_index - trojan_train - trojan_val] = matrix[source]
            trojan_index = trojan_index + 1
        elif result=='virus':
            if virus_index < virus_train:
                virus1[virus_index] = matrix[source]
            elif virus_index < virus_train + virus_val:
                virus2[virus_index - virus_train] = matrix[source]
            else:
                virus3[virus_index - virus_train - virus_val] = matrix[source]
            virus_index = virus_index + 1
        elif result=='worm':
            if worm_index < worm_train:
                worm1[worm_index] = matrix[source]
            elif worm_index < worm_train + worm_val:
                worm2[worm_index - worm_train] = matrix[source]
            else:
                worm3[worm_index - worm_train - worm_val] = matrix[source]
            worm_index = worm_index + 1
        else:
            if general_index < general_train:
                general1[general_index] = matrix[source]
            elif general_index < general_train + general_val:
                general2[general_index - general_train] = matrix[source]
            else:
                general3[general_index - general_train - general_val] = matrix[source]
            general_index = general_index + 1
        # print(trojan_index)
        # print(virus_index)
        # print(worm_index)
        
        # train_sample[sample_index] = matrix[source]
        # sample_index = sample_index + 1
        # result = Nodes[source].sample
        # result_list.append(result)
#pool = multiprocessing.Pool(processes=processors)
#pool.map(f,list(graph.keys()))
#print(type(list(graph.keys())))
#pool.map(f,sample_num)
#pool.close()
print(trojan_index)
print(virus_index)
print(worm_index)
print(general_index)
print(trojan1.shape)
print(virus1.shape)
print(worm1.shape)
print(general1.shape)
print(trojan2.shape)
print(virus2.shape)
print(worm2.shape)
print(general2.shape)
print(trojan3.shape)
print(virus3.shape)
print(worm3.shape)
print(general3.shape)
train_sample = np.r_[trojan1,virus1]
train_sample = np.r_[train_sample,worm1]
train_sample = np.r_[train_sample,general1]

print(train_sample.shape)
train_sample = np.r_[train_sample,trojan2]
train_sample = np.r_[train_sample,virus2]
train_sample = np.r_[train_sample,worm2]
train_sample = np.r_[train_sample,general2]
print(train_sample.shape)

train_sample = np.r_[train_sample,trojan3]
train_sample = np.r_[train_sample,virus3]
train_sample = np.r_[train_sample,worm3]
train_sample = np.r_[train_sample,general3]
print(train_sample.shape)
# print(graph)


delete_list = []
for node in sample_list:
    delete_list.append(node.num)

train_sample = np.delete(train_sample,delete_list,axis=1)
print(train_sample.shape)

np.save('samples_5000.npy',train_sample)

label = numpy.zeros(samples)

label_index = 0
# train
for i in range(label_index,label_index+trojan1):
    label[i] = 0
label_index = label_index + trojan1
for i in range(label_index,label_index+virus1):
    label[i] = 1
label_index = label_index + virus1
for i in range(label_index,label_index+worm1):
    label[i] = 2
label_index = label_index + worm1
for i in range(label_index,label_index+general1):
    label[i] = 3
label_index = label_index + general1



# val
for i in range(label_index,label_index+trojan2):
    label[i] = 0
label_index = label_index + trojan2
for i in range(label_index,label_index+virus2):
    label[i] = 1
label_index = label_index + virus2
for i in range(label_index,label_index+worm2):
    label[i] = 2
label_index = label_index + worm2
for i in range(label_index,label_index+general2):
    label[i] = 3
label_index = label_index + general2

# test
for i in range(label_index,label_index+trojan3):
    label[i] = 0
label_index = label_index + trojan3
for i in range(label_index,label_index+virus3):
    label[i] = 1
label_index = label_index + virus3
for i in range(label_index,label_index+worm3):
    label[i] = 2
label_index = label_index + worm3
for i in range(label_index,label_index+general3):
    label[i] = 3
label_index = label_index + general3

nums = label_index
diff_num = 0
print('\n\n\nTest Same:')
for i in range(nums):
    for j in range(nums):
        if train_sample[i] == train_sample[j] and label[i]!=label[j]:
            print(i,j,label[i],label[j])
            diff_num = diff_num+1
print('\n\n\nDiff Num:')
print(diff_num)
np.save('label.npy',label)


'''
l = len(Nodes)
# matrix = [[0] * len(Nodes) for i in range(len(Nodes))]
print(l)
print('--------')
print(graph)
# for i in range(len(Nodes)):
#     if Nodes[i].type_=='sample' or Nodes[i].type_=='process':
#         print(Nodes[i].type_,Nodes[i].name)


import networkx as nx
import numpy
Graph = nx.Graph()
# for node1 in range(len(Nodes)):
#     if node1 in graph:
#         for relation in graph[node1]:
#             for node2 in graph[node1][relation]:
#                 G.add_edge(node1,node2)
for node1 in graph:
    for relation in graph[node1]:
        for node2 in graph[node1][relation]:
            # print(node1,node2)
            # time.sleep(1)
            Graph.add_edge(node1,node2)

# matrix = numpy.zeros((l,l))
# for i in graph:
#     for type_i in graph[i]:
#         if type_i == 'sample':
#             for j in graph[i][type_i]:
#                 matrix[i][j] = 4
#         elif type_i == 'process':
#             for j in graph[i][type_i]:
#                 matrix[i][j] = 2
#         else:
#             for j in graph[i][type_i]:
#                 matrix[i][j] = 1

# 6.处理小图
# count = 0
# for sample in graph:
#     if Nodes[sample].type_=='sample':
#         for i in range(l):
#             if Nodes[i].type_=='sample' or Nodes[i].type_=='process':
#                 print(len(nx.shortest_path_length(G,sample,i)))
    # print(Nodes[sample].type_,Nodes[sample].name)
    # print('-----------')
    # print(Graph.__getitem__(sample))
    # count = count + 1
    # if count>10:
    #     break
'''
