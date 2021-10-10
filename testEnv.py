import numpy
import torch
import os
import json
from util.read_files import getAllFiles
# from util.get_label_id_by_hash import get_dict_of_hash_id
from util.mongo_func import read_file_hash_from_mongodb

InputDataPath = "../布谷鸟数据集/Win32_EXE/"


def judge_hash_of_json_and_mongo():
    # 通过hash得到_id， 需要考虑对应关系。
    malware_dict = read_file_hash_from_mongodb("192.168.105.224", "labels", "reportid_to_label_kind_name")
    jsonHash_dict = {}
    print(f'label num in mongo: {len(malware_dict)}')
    
    jsonFiles = getAllFiles(InputDataPath)
    for file in jsonFiles:
        f = open(file,'r')
        doc ={}
        doc = json.load(f)
        f.close()
        if 'sha256' in doc.keys():
            jsonHash_dict[doc['sha256']] = 1
            # if doc['sha256'] not in malware_dict:
            #     print(doc['sha256'])
            # else:
            #     print('-')
    print(f'label num in json files: {len(jsonHash_dict)}')


# malware_dict = read_file_hash_from_mongodb("192.168.105.224", "labels", "reportid_to_label_kind_name")
# for key in malware_dict:
#     print(key)
#     print(isinstance(key, str))
#     break

judge_hash_of_json_and_mongo()