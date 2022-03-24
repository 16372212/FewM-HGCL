# @Time: 2021.07.17
# @Author :ZhenZiyang

import json
import time
import os
import string
from pathlib import Path
import pandas as pd

InputDataPath = "../布谷鸟数据集/win32/"
OutputPath = "./label/label.csv"

NAME_LIST = {'troj': 'trojan', 'vir': 'virus', 'worm': 'worm', 'adw': 'adware', 'back': 'backdoor',
             'down': 'downloader', 'spy': 'spyware', 'drop': 'dropper', 'gen': 'general', 'gen_else': 'gen_else'}


def getAllFiles(analysis_root_dir):
    L = []
    for root, dirs, files in os.walk(analysis_root_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file))
    return L


def explainMicrosoftResult(result, file_name):
    """ 从类似Trojan:Win32/Malex.gen!F这样的格式中得到name, family"""
    name = ''
    family = ''
    list = result.split(':')

    if len(list) < 2:
        return '', '', ''
    # 规范化label的名字
    for key in NAME_LIST:
        if key in list[0]:
            name = NAME_LIST[key]
            break
    if name == '':
        return '', '', ''
    if '/' in list[1]:
        family_list = list[1].split('/')  # Win32/Skeeyah.A
        family = family_list[1]
    if '.' in family:
        family = family.split('.')[0]
    return name, family, file_name


def readScansFromMicrosoft(filename):
    """
    获取filename中，scan下Microsoft得到的结果：name+family

    基本格式：
        "Gen:Variant.Strictor.150188",
        "Trojan:Win32/Fuerboos.E!cl"
        "Trojan:Win32/VBClone"
        "Virus:Win32/Virut.BN"
    特殊情况：
        str(resultStr)的结果是'None'
        没有Microsoft这一项
    """

    f = open(filename, 'r')
    doc = {}
    results = {}
    # results: {"Microsoft": "Trojan:Win32/Fuerboos.E!cl", ...}
    doc = json.load(f)
    f.close()
    if 'scans' in doc.keys():
        # 判断是否有microsoft的结果
        if 'Microsoft' in doc['scans']:
            resultStr = doc['scans']['Microsoft']['result']
            # 保证结果不是None
            if str(resultStr).find('None') == -1:
                return explainMicrosoftResult(str(resultStr).lower(), doc['sha256'])
    return '', '', ''


def readScansFromAllCompany(filename):
    """
    获取filename中，scan下所有公司得到的结果：name+family
    通过投票机制选择name, familyName
    """
    f = open(filename, 'r')
    doc = {}
    results = {}
    # results: {"Microsoft": "Trojan:Win32/Fuerboos.E!cl", ...}
    doc = json.load(f)
    f.close()
    if ('scans' in doc.keys()):
        for key in doc['scans']:
            # key: Microsoft etc
            resultStr = doc['scans'][key]['result']
            if resultStr is not None:
                results[key] = resultStr.lower()
    return countFreqByVoting(results)


def genFamily(result, str):
    """
    默认name后面就是family。可进一步优化。
    """
    list = result.split('.')
    find = False
    for temp_str in list:
        if str in temp_str:
            find = True
            continue
        if find and temp_str != 'win32':
            return temp_str
    return ''


def countFreqByVoting(results):
    """
    投票机制选择结果。选择出得票最多的作为name。
    每个name又对应一个dict中的key, value存被预测得出的family的名称集（dict{trojin:[trojinFamilyname1, trojinFamilyname2,...]}）
    input: 
        results: {"Microsoft": "Trojan:Win32/Fuerboos.E!cl", ...}
    output: 
        投票后的name, familyName
    problem: 
        family记录时候容易有信息丢失

    """
    count = 0
    trojan = 0
    virus = 0
    worm = 0
    adware = 0
    backdoor = 0
    downloader = 0
    spyware = 0
    dropper = 0
    general = 0
    gen_else = 0

    name = ''
    family = ''

    max_freq = 0
    family_list = {'trojan': [], 'virus': [], 'worm': [], 'adware': [], 'backdoor': [],
                   'downloader': [], 'spyware': [], 'dropper': [], 'general': [], 'gen_else': []}

    if results == None:
        return name, family
    for key in results.keys():
        result = results[key]
        if 'trojan' in result or 'Trojan' in result or 'trj' in result or 'Trj' in result:
            trojan += 1
            if trojan > max_freq:
                max_freq = trojan
                name = 'trojan'
                family_name = genFamily(result, 'trj')
                if (family_name != ''):
                    family_list[name].append(family_name)

        elif 'vir' in result or 'Vir' in result:
            virus += 1
            if virus > max_freq:
                max_freq = virus
                name = 'virus'
                family_name = genFamily(result, 'vir')
                if (family_name != ''):
                    family_list[name].append(family_name)

        elif 'worm' in result or 'Worm' in result:
            worm += 1
            if worm > max_freq:
                max_freq = worm
                name = 'worm'
                family_name = genFamily(result, 'worm')
                if (family_name != ''):
                    family_list[name].append(family_name)

        elif 'Adw' in result or 'adw' in result or 'AdW' in result or 'adW' in result:
            adware += 1
            if adware > max_freq:
                max_freq = adware
                name = 'adware'
                family_name = genFamily(result, 'adw')
                if (family_name != ''):
                    family_list[name].append(family_name)

        elif 'back' in result or 'Back' in result:
            backdoor += 1
            if backdoor > max_freq:
                max_freq = backdoor
                name = 'backdoor'
                family_name = genFamily(result, 'back')
                if (family_name != ''):
                    family_list[name].append(family_name)

        elif 'spy' in result or 'Spy' in result:
            spyware += 1
            if spyware > max_freq:
                max_freq = spyware
                name = 'spyware'
                family_name = genFamily(result, 'spy')
                if (family_name != ''):
                    family_list[name].append(family_name)

        elif 'down' in result or 'Down' in result:
            downloader += 1
            if downloader > max_freq:
                max_freq = downloader
                name = 'downloader'
                family_name = genFamily(result, 'down')
                if (family_name != ''):
                    family_list[name].append(family_name)

        elif 'drop' in result or 'Drop' in result:
            dropper += 1
            if dropper > max_freq:
                max_freq = dropper
                name = 'dropper'
                family_name = genFamily(result, 'drop')
                if (family_name != ''):
                    family_list[name].append(family_name)

        elif 'gen' in result or 'Gen' in result:
            general += 1
            if general > max_freq:
                max_freq = general
                name = 'general'
                family_name = genFamily(result, 'gen')
                if (family_name != ''):
                    family_list[name].append(family_name)

    if (len(family_list[name]) > 0):
        return name, family_list[name][0]
    else:  # 需要被直接丢弃的数据
        return name, ''


def writeLableToCSV(labelList, path):
    dataframe = pd.DataFrame({'label': labelList})
    dataframe.to_csv(path, index=False, sep=',')


def getLableFromAllCompany(path):
    jsonFiles = getAllFiles(path)
    labels = []
    for file in jsonFiles:
        name, family = readScansFromAllCompany(file)
        labels.append(name + "." + family)
        print(name + "." + family)
    writeLableToCSV(labels, OutputPath)


def getLableFromMicrosoft(path):
    jsonFiles = getAllFiles(path)
    hash_dict = {}
    name_list = []
    family_list = []
    for file in jsonFiles:
        name, family, fhash = readScansFromMicrosoft(file)
        if name != '' and family != '':
            name_list.append(name)
            family_list.append(family)
            hash_dict[fhash] = {'name': name, 'family': family}
    return hash_dict, name_list, family_list


def getHashFromFiles(path):
    jsonFiles = getAllFiles(path)
    file_hash = []
    for filename in jsonFiles:

        f = open(filename, 'r')
        doc = {}
        doc = json.load(f)
        f.close()
        if ('sha256' in doc.keys()):
            file_hash.append(doc['sha256'])
    return file_hash


if __name__ == '__main__':
    """
    读取该目录下所有json文件，将name.FamilyName作为label，存成一个list
    采用投票机制。例如name “trojan”在多家公司中被预测的次数最多，该json就对应的name为“trojan”，
    相应的fFamilyName采用所有预测结果中，最后一个预测为“trojan”的结果对应的family name

    另一种办法：直接用Microsoft的结果：
        "Gen:Variant.Strictor.150188",
        "Trojan:Win32/Fuerboos.E!cl"
        "Trojan:Win32/VBClone"
        "Virus:Win32/Virut.BN"
    """
    # getLableFromAllCompany(InputDataPath)
    getLableFromMicrosoft(InputDataPath)
