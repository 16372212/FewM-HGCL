import os
import json
from util.get_name_family_from_file import getLableFromMicrosoft
from util.mongo_func import read_file_hash_from_mongodb, read_from_mongodb
import pymongo
from bson.objectid import ObjectId

InputDataPath = "../布谷鸟数据集/win32/"
OutputPath = "./label/label.csv"
host = "192.168.105.224"
port = 27017
dbname = "labels"
colname = "reportid_to_label_kind_name"
database_name = "cuckoo_nfs_db"


def writeLabelFromLocal():
    """读取目录下所有文件，写入mongo label中的一列
    写入过程中，需要mongo中读取hash对应的id
        1 get all file
        2 get list of name and family and hash
        3 read mongo， get id of which hash it holds
        4 write mongo: label, hash, id
    """
    # 1 从本地get all file, 得到file_hash, family, name:dict[file_hash: dict{'family', 'name'}]

    hash_dict = getLableFromMicrosoft(InputDataPath)

    # 2 从mongo中读取所有的dict{file_hash, _id}
    mongo_file_dict = read_file_hash_from_mongodb(host, dbname, colname)
    print(f'本地文档个数：{len(hash_dict)}')
    print(f'mongo文档个数：{len(mongo_file_dict)}')

    # label_dict = {}
    final_label_list = []

    lack_num = 0  # 本地路径下存在，但是在mongo中不存在的数目
    for file_hash in hash_dict:
        if file_hash in mongo_file_dict:
            id = mongo_file_dict[file_hash]['_id']
            # label_dict[id] = {'_id':id, 'file_hash':file_hash, 'family': hash_dict[file_hash]['family'], 'name': hash_dict[file_hash]['name']}
            final_label_list.append({'_id': id, 'file_hash': file_hash, 'family': hash_dict[file_hash]['family'],
                                     'name': hash_dict[file_hash]['name']})
        else:
            lack_num += 1
            # print("no this file_hash")
    print(f'处理后的个数{len(final_label_list)}')
    print(f'lack num{lack_num}')
    print(final_label_list[0])
    print(final_label_list[50])
    print(final_label_list[100])
    print(final_label_list[150])
    # 3 将得到dict['_id': dict{'file_hash', 'family', 'name', '_id'}]写入mongo
    # write_to_mongodb("192.168.105.224", "labels", "bugu_label_with_family", final_label_list)

    # 4 将family, name整合成embedding，然后写入到mongo


def readLabelFromMongo():
    label_dict = read_from_mongodb('192.168.105.224', 'labels', 'bugu_label_with_family')
    # label_dict[id]['family_name']
    # get list of name and family and hash
    # read mongo， get id of which hash it holds
    # write mongo: label, hash, id


call_id = {}


def readCalls():
    client = pymongo.MongoClient(host=host, port=port, unicode_decode_error_handler='ignore')

    call_collection = client['db_calls']['from_nfs_db1']
    collections = client[database_name]['analysis']

    # call_rows = call_collection.find(filter={'_id':ObjectId('5e1f134bdfa067752182eefd')})
    # call_rows = call_collection.find()

    # for call_row in call_rows:
    #     # print(call_row['_id'])
    #     if str(call_row['_id']) not in call_id:
    #         call_id[str(call_row['_id'])] = 1

    for row in collections.find():
        call_rows = call_collection.find(filter={'_id': row['_id']})
        for call_row in call_rows:
            print(call_row['_id'])
        # if str(row['_id']) in call_id:
        #     print(str(row['_id']))


# writeLabelFromLocal()
readCalls()
