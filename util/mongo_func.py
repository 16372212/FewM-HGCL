# -*- coding:utf-8 -*-
# @Author :TangMingyu

from pymongo import MongoClient
from bson.objectid import ObjectId

def read_from_mongodb(host, dbname, colname):
    port = 27017
    client = MongoClient(host, port, unicode_decode_error_handler='ignore')
    database = client[dbname]
    collection = database[colname]
    res = {}
    for line in collection.find():
        id = str(ObjectId(line['_id']))
        res[id] = line
    client.close()
    return res

def read_from_mongodb_ObjectId(host, dbname, colname):
    port = 27017
    client = MongoClient(host, port, unicode_decode_error_handler='ignore')
    database = client[dbname]
    collection = database[colname]
    res = {}
    for line in collection.find():
        id = str(line['_id'])
        res[id] = line
    client.close()
    return res


def write_to_mongodb(host, dbname, colname, data_list):
    port = 27017
    client = MongoClient(host, port, unicode_decode_error_handler='ignore')
    database = client[dbname]
    collection = database[colname]
    collection.insert_many(data_list)
    client.close()


def read_file_hash_from_mongodb(host, dbname, colname):
    port = 27017
    client = MongoClient(host, port, unicode_decode_error_handler='ignore')
    database = client[dbname]
    collection = database[colname]
    res = {}
    for line in collection.find():
        id = str(line['file_hash'])
        res[id] = line
    client.close()
    return res