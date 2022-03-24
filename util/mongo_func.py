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


def read_from_mongodb_objectid(host, dbname, colname):
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


def test_connect():
    port = 27017
    host = '192.168.105.224'
    dbname = 'cuckoo_nfs_db'
    coll_name = 'analysis'
    client = MongoClient(host, port, unicode_decode_error_handler='ignore')
    dblist = client.list_database_names()
    for db in dblist:
        print("database: ", db)

    database = client[dbname]
    collection = database[coll_name]
    print(f"collection {coll_name} in connect mongo count : {collection.count()}")


test_connect()