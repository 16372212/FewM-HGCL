from typing import List, Dict
from util.mongo_util import get_mongo_client
from util.label_util import get_labels_from_file
from util.const import host, databases_name, dbcalls_dict


def get_freq_of_api_comb() -> (List[int], List[Dict[str, int]]):
    labels = get_labels_from_file("../label/sample_result.txt")

    api_comb_list = get_api_combines()
    api_comb_freq_list = len(api_comb_list) * [0]

    api_freq_of_labels_list = [{'Trojan': 0, 'Worm': 0, 'Virus': 0, 'VirTool': 0, 'Backdoor': 0, 'SoftwareBundler': 0, 'DDoS': 0, 'PUA': 0, 'Ransom': 0, 'HackTool': 0, 'Program': 0, 'PWS': 0} for row in range(12)]

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
            file_hash = ''
            for row in rows:
                file_hash = row['file_hash']
            # 2. file_hash不存在就跳过
            if file_hash is None or file_hash not in labels:
                continue
            # 3. comb是否在对应的calls中, 并更新
            analyze_api_comb_freq(x['calls'], labels[file_hash]['label'], api_comb_list, api_comb_freq_list,
                                  api_freq_of_labels_list)
    return api_comb_freq_list, api_freq_of_labels_list


def analyze_api_comb_freq(calls: List[str], label: str, api_comb_list: List[List[str]], api_freq_list: List[int],
                          api_freq_of_labels_list: List[Dict[str, int]]):
    index = 0
    for api_comb in api_comb_list:
        match = True
        for api in api_comb:
            if api not in calls:
                match = False
                break
        if match:
            api_freq_list[index] += 1
            if label not in api_freq_of_labels_list[index]:
                api_freq_of_labels_list[index][label] = 1
            else:
                api_freq_of_labels_list[index][label] += 1
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
    api_freq_list, api_freq_of_labels = get_freq_of_api_comb()
    # [0, 0, 0, 0, 6, 0, 267, 3388, 0, 0, 19, 0]
    i = 0
    for freq in api_freq_of_labels:
        i += 1
        print(f'combine{i}: {freq}')
        print(f'total sum of combing: {api_freq_list[i]}')

