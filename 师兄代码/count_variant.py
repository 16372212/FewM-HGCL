# @Author :SuMing

import json
import time
import os
f = open('result_all.txt','r')
docs ={}
results = {}
results = json.load(f)
f.close()
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
for sample in results:
    count += 1
    for scan in results[sample]:
        result = results[sample][scan]
        if result==None:
            continue
        if 'variant' in result or 'Variant' in result:
            if 'trojan' in result or 'Trojan' in result or 'trj' in result or 'Trj' in result:
                trojan += 1
                break
            elif 'vir' in result or 'Vir' in result:
                virus += 1
                break
            elif 'worm' in result or 'Worm' in result:
                worm += 1
                break
            elif 'Adw' in result or 'adw' in result or 'AdW' in result or 'adW' in result:
                adware += 1
                break
            elif 'back' in result or 'Back' in result:
                backdoor += 1
                break
            elif 'spy' in result or 'Spy' in result:
                spyware += 1
                break
            elif 'down' in result or 'Down' in result:
                downloader += 1
                break
            elif 'drop' in result or 'Drop' in result:
                dropper += 1
                break
            elif 'gen' in result or 'Gen' in result:
                general += 1
                break
        # print(results[sample][scan])
        # if count%5==0:
            # break
    if count%10000==0:
        print('count:',count)
        print(trojan + virus + worm + adware + backdoor + downloader + spyware + dropper)
        print(general)
        # print(trojan)
        # print(virus)
        # print(worm)
        # print(adware)
        # print(backdoor)
        # print(downloader)
        # print(spyware)
        # print(dropper)
print('Count:',count)
print('Result:')
print(trojan)
print(virus)
print(worm)
print(adware)
print(backdoor)
print(downloader)
print(spyware)
print(dropper)
print(general)
# print(len(docs))

# keys = docs.keys()
# doc2 = {}
# print(type(keys))
# count = 0
# for name in keys:
#   count += 1
#   if count%500 == 0:
#       print(count)
#   file_path = ''
#   file_path1 = "D:\cuckoo\cuckoo\win32\Win32_EXE\\"+str(name)+'.json'
#   file_path2 = "D:\cuckoo\cuckoo\win32_45\Win32_EXE\\"+str(name)+'.json'
#   if os.path.exists(file_path1)==True:
#       file_path = file_path1
#   elif os.path.exists(file_path2)==True:
#       file_path = file_path2
#   else:
#       continue
#   f = open(file_path,'r')
#   results = json.load(f)
#   #print(results)
#   results_key = results['scans'].keys()
#   tmp = {}
#   for result_key in results_key:
#       value = results['scans'][result_key]['result']
#       tmp[result_key] = value
#   doc2[name] = tmp


#   # print(results)
# f = open("result_all_2.txt",'w+')
# json.dump(doc2,f)
# # break




