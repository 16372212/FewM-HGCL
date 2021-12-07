import os
import json
<<<<<<< HEAD
# from util.get_name_family_from_file import getLableFromMicrosoft, getHashFromFiles
=======
from util.get_name_family_from_file import getLableFromMicrosoft, getHashFromFiles
>>>>>>> af4f17e (add gcc)

InputDataPath = "../../布谷鸟数据集/win32/"
OutputPath = "../label/label.csv"


<<<<<<< HEAD
# def analyze_file_hash():
#     hash_file = getHashFromFiles(InputDataPath)
#     print(hash_file)
#
#
# def old_analyze_name_family():
#
#     hash_dict, name_list, family_list = getLableFromMicrosoft(InputDataPath) # [hash]: {'name': name, 'family':family}
#
#
#     # name: different family set()
#     family_dict = {}
#     for hashs in hash_dict:
#         if hash_dict[hashs]['name'] not in family_dict:
#             family_dict[hash_dict[hashs]['name']] = set()
#         family_dict[hash_dict[hashs]['name']].add(hash_dict[hashs]['family'])
#
#     print('====================number of all labels=========================')
#     print(len(hash_dict))
#
#     print('====================len of name and its family num=========================')
#     print(len(family_dict))
#
#     for names in family_dict:
#         print(f'{names}: {len(family_dict[names])}')
#
#     print(set(name_list))
#
#     print('====================family set of all names=========================')
#     for names in family_dict:
#         print(f'{names}: {family_dict[names]}')
#         print()
#
#     print('所以可以看到这种策略下，存在mira!rfh, 也存在mira，那么需要区分这两种family吗？还是直接都按照mira来计算')
#
#     return family_dict


# def old_mapFamilyAndName2id():
#     family_dict = old_analyze_name_family()
#     i = 0
#     id_dict = {}
#     for names in family_dict:
#         for family in family_dict[names]:
#             id_dict[names+family] = i
#             i+=1
#     print(id_dict)
#     return id_dict


def analyze_familys():
    family_samples_num = {}  # {family_name: num}
    f = open('../label/sample_result.txt', 'r')
    datas = f.read().split('\n')
    labels = {}
    for data in datas:
        if ',' in data:
            data = data.split(',')
            file_hash = data[0]
            type1 = data[1]
            type2 = data[2]
            labels[file_hash] = {}
            labels[file_hash]['label'] = type1
            labels[file_hash]['family'] = type2
            if type2 in family_samples_num:
                family_samples_num[type2] += 1
            else:
                family_samples_num[type2] = 1
    f.close()
    a = sorted(family_samples_num.items(), key=lambda x: x[1], reverse=True)
    print(a)
    print('Loaded Labels')
    print(f'total family nums: {len(family_samples_num)}')
    for fam in family_samples_num:
        print(f'{fam}: {family_samples_num[fam]}')



analyze_familys()
=======
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
>>>>>>> af4f17e (add gcc)
