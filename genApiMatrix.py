# @Author :ZhenZiyang

import pymongo
from bson import json_util as jsonb
from util.mongo_func import read_from_mongodb
import json
import time
import os
import re
import numpy as np
import numpy.ma as ma
import pickle

api_index_map = {}

api_matrix = [] # {name: , category: , Arguments: (registory)(一个大字符串，开头格式须一致)}

databases_name = ["cuckoo_nfs_db"]

api_name_word = set()
api_name_word_dict = {}

api_category_letter = set()
api_category_letter_dict = {}

api_argu_word = set()
api_argu_word_dict = {}

NAME_SIZE = 8
CATEGORY_SIZE = 4
ARGU_SIZE = 20

OUTPUT_PATH = "./mid_data/api_matrix.pkl"

ip = "192.168.105.224"
port = 27017
collection_name = "calls"


def feature_hashing(word, index, size):
    num = index % size
    ret = np.zeros(size, dtype=np.int32, )
    # ret第num维应该赋值为1
    ret[num] = 1
    return ret


def set_api_word_dict(name, size, api_dict, api_set):
    """驼峰转单词集合，
        完善dict, set
        返回众多feature_hashing集合的并"""
    
    word_list = re.findall('[A-Z][^A-Z]*', name)
    # print('word list: ')
    # print(word_list)
    ret_vec = np.zeros(size, dtype=np.int32, )
    for word in word_list:
        if word not in api_set:
            api_dict[word] = len(api_set)
            api_set.add(word)
        ret_vec += feature_hashing(word, api_dict[word], size)
    # print(f'ret_vec is {ret_vec}')
    return ret_vec


def set_argu_dict(argument_dict):
    """判断是否有regkey, 且如果开头是HKEY，则获取分割后的值。将这些值分别加入dict"""
    ret_vec = np.zeros(ARGU_SIZE, dtype=np.int32, )
    if 'regkey' in argument_dict:
        if argument_dict['regkey'].startswith('HKEY'):
            # 分割字符串，
            argu_list = argument_dict['regkey'].split('\\')
            # print('argu list: ')
            # print(argu_list)
            for args in argu_list:
                if args not in api_argu_word:
                    api_argu_word_dict[args] = len(api_argu_word)
                    api_argu_word.add(args)
                ret_vec += feature_hashing(args, api_argu_word_dict[args], ARGU_SIZE)
    # print(f'argu ret_vec is {ret_vec}')
    return ret_vec


def readApiFromMongo():
    
    for database_name in databases_name:
        client = pymongo.MongoClient(host=ip, port=port,unicode_decode_error_handler='ignore')
        bad_collections = client[database_name][collection_name]

        print(f'----------------begin to write database {database_name}---------------')
        print(f'len of api_matrix is {len(api_matrix)}')

        for x in bad_collections.find():
            if 'calls' not in x:
                continue
            calls = x['calls']
            for call in calls:

                
                # print(call['category'])
                # print(call['arguments'])

                if call is None or 'api' not in call:
                    continue
                

                api_name = call['api']
                if api_name in api_index_map:
                    continue

                print(call['api'])
                if api_name == 'NtDeleteFile':
                    print('choose to break')
                    break

                api_index_map[call['api']] = len(api_matrix)
                
                array_name = set_api_word_dict(call['api'], NAME_SIZE, api_name_word_dict, api_name_word) # 完善dict, set, 并返回对应的向量并的结果. name, size, api_dict, api_set
                array_category = set_api_word_dict(call['category'].upper(),CATEGORY_SIZE , api_category_letter_dict, api_category_letter) # 根据每个字母进行分割
                array_arguments = set_argu_dict(call['arguments']) # 判断一开始是否有HKEY开头
                
                total_vec = np.hstack((array_name, array_category, array_arguments))

                api_matrix.append(total_vec)

                # 这里发现，相同api对应得，argu会有变化，但是api name, category没有变化。
                # if call['api'] not in api_index_map:
                #     api_index_map[call['api']] = []
                # if len(api_index_map[call['api']]) != 0 and all(api_index_map[call['api']][0] == total_vec) :
                #     i = 1
                # else:
                #     api_index_map[call['api']].append(total_vec)
        if len(api_matrix) > 341:
            break

    print(len(api_matrix))

    with open(OUTPUT_PATH, 'wb') as fr:
        pickle.dump(api_matrix, fr)

        pickle.dump(api_index_map,fr)


readApiFromMongo()

def readApiFromFile():

    with open(OUTPUT_PATH, 'rb') as fr:
        api_matrix = pickle.load(fr)
        api_index_map = pickle.load(fr)


def maskApiMatrix(api_matrix):

    randome_mask_location = np.random.randint(0,2,(api_matrix.shape[0], api_matrix.shape[1]))
    randome_mask_value = np.random.normal(size=(4,4)) # 生成高斯分布的矩阵，这里需要找到所有一维矩阵的的最大值和最小值
    return api_matrix*(1-randome_mask) + randome_mask_value*randome_mask

    # api_mask_matrix = ma.masked_array(api_matrix, mask=randome_mask)

