import io
import os 



def read_files(file_dir, root):

    write_file_path = 'total_result.txt'
    with open(write_file_path, mode='a', encoding='utf-8') as file_obj:    
        
        dir_list = os.listdir(file_dir)
        for cur_file in dir_list:
            # 获取文件的绝对路径
            path = os.path.join(file_dir, cur_file)
            if os.path.isdir(path) and cur_file.startswith("Pretrain"):
                read_files(path,2) # 递归子目录
            elif root == 2 and cur_file=="result.txt":
                # print(file_dir)
                file_list = file_dir.split('_')
                layer_index = file_list.index("layer") + 1
                bsz_index = file_list.index("bsz") + 1
                lamda = file_list.index("momentum") + 1
                r_index = file_list.index("momentum") + 2   
                file_obj.write("layer: " + file_list[layer_index] + " batch_size: " + file_list[bsz_index] + " lamda: " + file_list[lamda] + " " + file_list[r_index] + "\n")
                # read from this path, and print to a txt
                with open(path,"r") as f:    #设置文件对象
                    str = f.read()    #可以是随便对文件的操作
                    file_obj.write(str)
                file_obj.write('\n')
                file_obj.write('\n')

read_files('./', 1)