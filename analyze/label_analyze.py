from typing import Dict
from util.label_util import get_labels_from_file, get_labels_from_mongo


def sum_of_labels_and_families(label: Dict[str, Dict[str, str]]):
    """这里总结的所有的数据里的family、label的个数"""
    print(f'total sample num: {len(label)}')
    label_dict: Dict[str, int] = {}
    family_dict: Dict[str, int] = {}
    for each_hash in label:
        if label[each_hash]['label'] not in label_dict:
            label_dict[label[each_hash]['label']] = 1
        else:
            label_dict[label[each_hash]['label']] += 1
        if label[each_hash]['family'] not in family_dict:
            family_dict[label[each_hash]['family']] = 1
        else:
            family_dict[label[each_hash]['family']] += 1
    print(f'total label category : {len(label_dict)} \n')
    print(f'total family category : {len(family_dict)} \n')
    print(label_dict)
    print(family_dict)


if __name__ == "__main__":
    labels: Dict[str, Dict[str, str]] = get_labels_from_file("../label/sample_result.txt")
    sum_of_labels_and_families(labels)
    sample_labels: Dict[str, Dict[str, str]] = get_labels_from_mongo(labels)
    # print("label of initial files")
    # sum_of_labels_and_families(labels)
    # print("label of mongo")
    # sum_of_labels_and_families(sample_labels)
    # print("family of initial files")
    # sum_of_labels_and_families(labels)
    # print("family of mongo")
    # sum_of_labels_and_families(sample_labels)
