# 异构图构建
布谷鸟异构图构建，对比学习

## 项目结构介绍

dgl_compare_learning:

- utl 一些可复用性功能
- demo_test 一些test demo, 测试数据库的链接性、文件是否存在、以及简单例子测试
- train 
  - prepare_gcc_data
  - GCC gcc作为模型进行测试
- data （data部分数据暂时只上传input部分，其他需在gitignore中声明）
  - input_data: input data
  - mid_data: 中间生成的数据
  - out_data: output data
- prepare 数据预处理
  - read_data 读取数据
  - draw_graph
  - build_dgl_from_graph
  - prepare.py
- analyze 统计类的工作
- 资料 纸质版实验资料


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



数据集：

布谷鸟数据集

# 数据集说明

布谷鸟数据集整理后的放在了mongoDB中。

cuckoo_nfs_dX中的数据里，calls这个collection不是和analysis以及其他对应的。
