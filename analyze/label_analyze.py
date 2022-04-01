from typing import Dict
from util.label_util import get_labels_from_file, get_labels_from_mongo


def sum_of_labels_and_families(label: Dict[str, Dict[str, str]]):
    """这里总结的所有的数据里的family、label的个数"""
    print(f'total sample num: {len(label)}')
    label_dict: Dict[str, int] = {}
    family_dict: Dict[str, int] = {}
    label_family_dict: Dict[str, Dict[str, int]] = {}  # label: contain families
    for each_hash in label:
        big_category_name = label[each_hash]['label']
        small_category_name = label[each_hash]['family']
        if big_category_name not in label_dict:
            label_dict[big_category_name] = 1
        else:
            label_dict[big_category_name] += 1
        if small_category_name not in family_dict:
            family_dict[small_category_name] = 1
        else:
            family_dict[small_category_name] += 1

        # 计算每个大类对应有多少种小类
        if big_category_name not in label_family_dict:
            label_family_dict[big_category_name] = {}
        if small_category_name not in label_family_dict[big_category_name]:
            label_family_dict[big_category_name][small_category_name] = 1
        else:
            label_family_dict[big_category_name][small_category_name] += 1

    print(f'total label category : {len(label_dict)} \n')
    print(f'total family category : {len(family_dict)} \n')
    print(label_dict)
    print(family_dict)
    print(label_family_dict)

    print()
    i = 1
    for label in label_family_dict:
        print(f'{i}. for label {label}:')
        print(f'there are {len(label_family_dict[label])} families')
        print(label_family_dict[label])
        print()
        i += 1


if __name__ == "__main__":
    labels: Dict[str, Dict[str, str]] = get_labels_from_file("../label/sample_result.txt")
    sum_of_labels_and_families(labels)
    # total sample num: 50434
    # sample_labels: Dict[str, Dict[str, str]] = get_labels_from_mongo(labels)
    # print("label of initial files")
    # sum_of_labels_and_families(labels)
    # print("label of mongo")
    # sum_of_labels_and_families(sample_labels)
    # print("family of initial files")
    # sum_of_labels_and_families(labels)
    # print("family of mongo")
    # sum_of_labels_and_families(sample_labels)
