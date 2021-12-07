# 异构图构建
布谷鸟异构图构建，对比学习

## 运行
### 1. 先生成需要的graph, aug_graph数据
aug_graph的生成: 运行matrix_to_huge_dgl.py, 得到gcc\gen_my_datasets\aug_graphs_x

修改my_graph_dataset.py中aug_graphs_x数据的路径。

### 2. 训练
gcc文件夹下是根据gcc的源码改写内容，任务可分成节点分类、图分类。

本次实现方法是图分类。

## 运行方法：sh run.sh 对比实验：sh run_loop.sh

## 实验结果
result.rar文件夹下的result.txt可见结果。打印出所有result文件夹下的txt集合后，结果在total_result.txt中。 最高准确率（108类）达到0.81，但大类分类结果比这个低

#### 原因

big_label这里处理的有些问题。可以修改matrix_to_huge_dgl.py， my_
