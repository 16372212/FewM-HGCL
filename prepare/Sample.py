import os

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
        #  num 编号, name 唯一标识, type_ 种类, sample 归属样本, pid 进程号, 后两个没用
        ## type_: process, file, reg, memory, sign, network
        ## process: x['behavior']['processtree'] to do dfs
        ## file   : x['behavior']['generic'][(数组下标)]['summary'][api] , api in ['file_created','file_written']
        ## reg    : x['behavior']['generic'][(数组下标)]['summary'][api] , api in ['regkey_read','regkey_opened'] 提取 {} 部分
        ## memory : x['behavior']['generic'][(数组下标)]['summary']['dll_loaded']
        ## sign   : x['signatures'][(数组下标)]['name']
        ## network: x['network']['tcp']['dst'] | x['network']['udp']['dst']
        ##
