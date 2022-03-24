import os
import json
from util.get_name_family_from_file import getLableFromMicrosoft, getHashFromFiles

InputDataPath = "../../布谷鸟数据集/win32/"
OutputPath = "../label/label.csv"


def analyze_file_hash():
    hash_file = getHashFromFiles(InputDataPath)
    print(hash_file)


def analyze_name_family():

    hash_dict, name_list, family_list = getLableFromMicrosoft(InputDataPath) # [hash]: {'name': name, 'family':family}

    # name: different family set()
    family_dict = {}

    for hashs in hash_dict:
        if hash_dict[hashs]['name'] not in family_dict:
            family_dict[hash_dict[hashs]['name']] = set()
        family_dict[hash_dict[hashs]['name']].add(hash_dict[hashs]['family'])

    print('====================number of all labels=========================')
    print(len(hash_dict))
    
    print('====================len of name and its family num=========================')
    print(len(family_dict))
    
    for names in family_dict:
        print(f'{names}: {len(family_dict[names])}')

    print(set(name_list))

    print('====================family set of all names=========================')
    for names in family_dict:
        print(f'{names}: {family_dict[names]}')
        print()

    print('所以可以看到这种策略下，存在mira!rfh, 也存在mira，那么需要区分这两种family吗？还是直接都按照mira来计算')

    return family_dict


def mapFamilyAndName2id():
    family_dict = analyze_name_family()
    i = 0
    id_dict = {}
    for names in family_dict:
        for family in family_dict[names]:
            id_dict[names+family] = i
            i+=1
    print(id_dict)
    return id_dict


mapFamilyAndName2id()

