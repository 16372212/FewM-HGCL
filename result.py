# @Time: 2021.07.17
# @Author :ZhenZiyang

import json
import time
import os
f = open('api_count_all.txt','r')
docs ={}
results = {}
docs = json.load(f)
f.close()
print(len(docs))

keys = docs.keys()
doc2 = {}
print(type(keys))
count = 0
for name in keys:
	count += 1
	if count%500 == 0:
		print(count)
	file_path = ''
	file_path1 = "D:\cuckoo\cuckoo\win32\Win32_EXE\\"+str(name)+'.json'
	file_path2 = "D:\cuckoo\cuckoo\win32_45\Win32_EXE\\"+str(name)+'.json'
	if os.path.exists(file_path1)==True:
		file_path = file_path1
	elif os.path.exists(file_path2)==True:
		file_path = file_path2
	else:
		continue
	f = open(file_path,'r')
	results = json.load(f)
	#print(results)
	results_key = results['scans'].keys()
	tmp = {}
	for result_key in results_key:
		value = results['scans'][result_key]['result']
		tmp[result_key] = value
	doc2[name] = tmp


	# print(results)
f = open("result_all_2.txt",'w+')
json.dump(doc2,f)
#	break




