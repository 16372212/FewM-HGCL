# 数据集标注

## 需求：
标准的数据的格式是：`worm.win32.family....`

在这些获取的文件中，格式并不完整，或者并不是按照这个类型出现的。现在需要将每个文件的标签，从`worm`变为`family.worm`

数据集标注

- 考虑到格式问题，直接用microsoft的结果

- 将label更新标注后，写入到mongo中：格式：id : file_hash : label_name。file_hash这里有一个转换。找到仅存的hash，找到相应的id, 再搞lebel_name



## 步骤
### behavior与id的对应

1. 解析labels，写入mongo


```python
# get_two_matrix.act_dataset_api_calls_to_ids里
id = str(ObjectId(x['_id']))  # 5e2191a666961368c8596eac

def act_dataset_malware_reportid_list_to_labels(reportid_list):
   malware_dict = read_from_mongodb("192.168.105.224", "labels", "reportid_to_label_kind_name")
   label_name_list = []
   for reportid in reportid_list:
      if reportid in malware_dict:
         label_name_list.append(label_name_id_dict[malware_dict[reportid]['label_name']])
      else:
         label_name_list.append(7)
   return label_name_list

```
是文件名称作为hash，id是... 

还没找到是怎么对应的

2. 根据新写入的mongo列，读取对应的特征内容。

3. 根据新的matrix构图

4. 数据增强


11月末交，所以12号之前必须初步掌握构图策略（解决完数据的对应问题），16号前构图结束。20号进行完增强实验。10月末开始训练