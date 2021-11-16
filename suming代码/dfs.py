## 深搜遍历，用于连接样本、进程
## 逻辑： sample 和 子process连接, 子process 之间相互可能连接
def dfs(process,sample):
    process_name = process['process_name'].replace(' .','.')
    if process_name=='cmd.exe':
        return None
    current = ''
    # 当前process 是样本本身
    if sample.name in process_name:
        current = sample
    # 当前process 是子进程
    else:
        if process_name not in process_list:
            process_list.append(process_name)
            process_map[process_name] = len(Nodes)
            pronode = Node(len(Nodes),process_name,'process','',0)
            Nodes.append(pronode)
        else:
            # pronode 赋值为 原有已建立好的节点
            pronode = Nodes[process_map[process_name]]
        current = pronode
        connect(sample,pronode) # 为啥这个要链接

    if 'children' in process:
        # dfs遍历所有子节点
        for children in process['children']:
            childnode = dfs(children,sample)
            # 如果子节点不是sample样本本身，则建立新的连接
            if childnode:
                # connect(current,childnode)
                # ******** 适当修改
                connect(sample,childnode)
                
    # 若当前节点是样本本身，则不返回内容，以免重复连接
    if current != sample:
        return current
    return None
