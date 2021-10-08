import os
import json
from util.get_name_family_from_file import getLableFromMicrosoft

"""读取目录下所有文件，写入mongo label中的一列
写入过程中，需要mongo中读取hash对应的id
    1 get all file
    2 get list of name and family and hash
    3 read mongo， get id of which hash it holds
    4 write mongo: label, hash, id
"""
def writeLable():
    # get all file, 暂时写入本地

    InputDataPath = "../布谷鸟数据集/Win32_EXE/"
    OutputPath = "./label/label.csv"
    name_list, family_list = getLableFromMicrosoft(InputDataPath)

    # get list of name and family and hash
    # read mongo， get id of which hash it holds
    # write mongo: label, hash, id


