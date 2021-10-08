import os
import json
from util.get_name_family_from_file import getLableFromMicrosoft, getHashFromFiles

InputDataPath = "../布谷鸟数据集/win32/"
OutputPath = "./label/label.csv"


def analyze_file_hash():
    hash_file = getHashFromFiles(InputDataPath)
    print(hash_file)


def analyze_name_family():

    name_list, family_list = getLableFromMicrosoft(InputDataPath)

    print('====================len of name and family=========================')
    print(len(name_list))
    print(len(family_list))

    name_set = set(name_list)
    family_set = set(family_list)

    print('====================set of name and family=========================')
    print(name_set)
    print(family_set)

    name_dict = {}

    i = 0
    for name in name_list:
        if name not in name_dict:
            name_dict[name] = []
        name_dict[name].append(family_list[i])  
        i += 1


    print('====================dict:\{name : family_name\}=========================')

    for name in name_dict:
        print(f'name:{name}, number of name:{len(name_dict[name])}')
        temp_family_set = set(name_dict[name])
        print(temp_family_set)
        print()


analyze_file_hash()