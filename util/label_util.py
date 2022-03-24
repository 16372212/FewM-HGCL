from typing import Dict
from util.mongo_util import get_mongo_client
from util.const import host, databases_name


def get_labels_from_file(filename: str) -> Dict[str, Dict[str, str]]:
    f = open(filename, 'r')
    all_data = f.read().split('\n')
    labels: Dict[str, Dict[str, str]] = {}
    for data in all_data:
        if ',' in data:
            data = data.split(',')
            file_hash = data[0]
            type1 = data[1]
            type2 = data[2]
            labels[file_hash] = {}
            labels[file_hash]['label'] = type1
            labels[file_hash]['family'] = type2
    f.close()
    return labels


def get_labels_from_mongo(labels: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """从mongo中的db中读取到每个"""
    labels_from_mongo: Dict[str, Dict[str, str]] = {}
    for database_name in databases_name:
        print(f"getting label from database: {database_name}")
        client = get_mongo_client(host)
        collections = client[database_name]['analysis']
        file_collection = client[database_name]['report_id_to_file']
        # api_collection = client['db_calls'][dbcalls_dict[database_name]]
        cursor = collections.find(no_cursor_timeout=True)
        for x in cursor:
            # 进程list,包括样本
            # 获取hash
            rows = file_collection.find(filter={'_id': str(x['_id'])})
            for row in rows:
                file_hash = row['file_hash']
                if file_hash is None or file_hash not in labels:
                    continue
                labels_from_mongo[file_hash] = {}
                labels_from_mongo[file_hash]['label'] = labels[file_hash]['label']
                labels_from_mongo[file_hash]['family'] = labels[file_hash]['family']
            if len(labels_from_mongo) % 100 == 0:
                print(f"total sum of labels: {len(labels_from_mongo)}")
        client.close()
    return labels_from_mongo
