from typing import List
from util.mongo_util import get_mongo_client
from util.label_util import get_labels_from_file
from util.const import host, databases_name, dbcalls_dict


def get_freq_of_api_comb() -> List[int]:
    labels = get_labels_from_file("../label/sample_result.txt")
    api_comb_list = get_api_combines()
    api_comb_freq_list = len(api_comb_list) * [0]
    client = get_mongo_client(host)
    # 遍历所有的calls类
    for database_name in databases_name:
        print(f"database: {database_name}")
        call_collection = client['db_calls'][dbcalls_dict[database_name]]
        # ps: call_collection中的id并不是直接对应的db_calls或analyze中的ID，需要使用file_collection转化
        file_collection = client[database_name]['report_id_to_file']
        cursor = call_collection.find(no_cursor_timeout=True)
        for x in cursor:
            # 1. 获取映射到的file_hash
            rows = file_collection.find(filter={'_id': str(x['_id'])})
            for row in rows:
                file_hash = row['file_hash']
            # 2. file_hash不存在就跳过
            if file_hash is None or file_hash not in labels:
                continue
            # 3. comb是否在对应的calls中
            calls = x['calls']
            # print(calls)
            # 判断calls中是否包含某种组合
            analyze_api_comb_freq(calls, api_comb_list, api_comb_freq_list)
    return api_comb_freq_list


def analyze_api_comb_freq(calls: List[str], api_comb_list: List[List[str]], api_comb_freq_list: List[int]):
    index = 0
    for api_comb in api_comb_list:
        match = True
        for api in api_comb:
            if api not in calls:
                match = False
                break
        if match:
            api_comb_freq_list[index] += 1
        index += 1


def get_api_combines() -> List[List[str]]:
    api_comb_list: List[List[str]] = [["NtReadFile", "CDocument_Write"],
                                      ["System", "NtAllocateVirtualMemory"],
                                      ["Process32FirstW", "Process32Next"],
                                      ["GetFileType", "GetFileSize", "IdrLoadDll"],
                                      ["EnumServicesStatusW", "GetSystemTimeAsFileTime"],
                                      ["SetEndOfFile", "IdrGetDllHandle"],
                                      ["NtEnumerateKey", "FindResourceExW"],
                                      ["GetVolumePathNamesForVolumeNameW", "RegQueryValueExW"],
                                      ["GetCurSorPos", "InternetOpenA", "RegOpenKeyExW"],
                                      ["EnumWindows", "GetCurSorPos", "GetAddrInfoW"],
                                      ["GetSystemMetrics", "SendNotifyMessageA"],
                                      ["RemoveDirectoryW", "OutputDebugStringA", "CouninItialize"]]
    return api_comb_list


if __name__ == "__main__":
    api_comb_freq_list = get_freq_of_api_comb()
    print(api_comb_freq_list)
