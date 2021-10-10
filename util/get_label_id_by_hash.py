# not used

def act_dataset_malware_reportid_list_to_labels(reportid_list):
    """根据id list得到label中的name。返回name_list"""
    malware_dict = read_file_hash_from_mongodb("192.168.105.224", "labels", "reportid_to_label_kind_name")
    label_name_list = []
    for reportid in reportid_list:
        if reportid in malware_dict:
            label_name_list.append(label_name_id_dict[malware_dict[reportid]['label_name']])
        else:
            label_name_list.append(7)
    return label_name_list




