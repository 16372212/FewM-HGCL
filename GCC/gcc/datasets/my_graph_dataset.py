#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/11 12:17
# TODO:

import math
import operator
import logging
import dgl
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from dgl.data import AmazonCoBuy, Coauthor
import os, sys
root_path = os.path.abspath("./")
sys.path.append(root_path)
from gcc.datasets import data_util
from dgl.data.utils import load_graphs
import scipy.sparse as sparse
from scipy.sparse import linalg
import sklearn.preprocessing as preprocessing
import torch.nn.functional as F


GRAPH_SUB_AUG_INPUT_PATH = 'gcc/gen_my_datasets/aug_graphs_15/aug_'


def add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed(worker_info.seed % (2 ** 32))

class GraphDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        super(GraphDataset).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        #  graphs = []
        graphs, _ = dgl.data.utils.load_graphs(
            "data_bin/dgl/lscc_graphs.bin", [0, 1, 2]
        )
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])

    def __len__(self):
        return self.length

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        # 针对图分类，id是针对图的id
        graph_idx, node_idx = self._convert_idx(idx)

        # 这里q是构造的正样本。
        graph_q = data_util._my_aug_for_dgl(
            g=self.graphs[graph_idx],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )

        # k是原图
        graph_k = data_util._my_ori_dgl(
            g=self.graphs[graph_idx],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=True,
        )
        return graph_q, graph_k


class NodeClassificationDataset(GraphDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert positional_embedding_size > 1

        self.data = data_util.create_node_classification_dataset(dataset).data
        self.graphs = [self._create_dgl_graph(self.data)]
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

    def _create_dgl_graph(self, data):
        graph = dgl.DGLGraph()
        src, dst = data.edge_index.tolist()
        num_nodes = data.edge_index.max() + 1
        graph.add_nodes(num_nodes)
        graph.add_edges(src, dst)
        graph.add_edges(dst, src)
        graph.readonly()
        return graph


class MyGraphClassificationDataset(NodeClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.entire_graph = True
        assert positional_embedding_size > 1

        # TODO 修改自己的dataset作为一部分
        self.dataset = data_util.create_graph_classification_dataset()
        self.k_graphs = self.dataset['graph_k_lists']  # 原本子图
        self.q_to_k_index_dict = self.dataset['q_to_k_index']
        self.length = self.dataset['k_qnum'][-1]
        self.total = self.length
        self.k_qnum = self.dataset['k_qnum']
        print(f'=====k_graph_size:{len(self.k_graphs)}, q_graph_size:{self.length}======')

    def __getitem__(self, idx):
        # 针对图分类，id是针对图的id
        
        k_idx = int(self.dataset['q_to_k_index'][idx])
        q_idx = int(idx - self.dataset['k_qnum'][k_idx])

        model_path = GRAPH_SUB_AUG_INPUT_PATH+str(k_idx)+'.bin'
        graph_q_set = load_graphs(model_path)[0]

        graph_k = self.k_graphs[k_idx]
        graph_q = graph_q_set[q_idx]
        # print(f'--------------------{idx} :/ {k_idx}---------------------------')
        # return graph_q, graph_kd
        # gq = add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
        # gk = add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)
        return graph_k, graph_q


class MyGraphClassificationDatasetForOnlyOrigin(MyGraphClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.entire_graph = True
        assert positional_embedding_size > 1

        self.dataset = data_util.create_graph_classification_dataset()
        self.k_graphs = self.dataset['graph_k_lists']  # 原本子图
        self.q_graphs = self.dataset['graph_q_lists']  # 增强的子图
        self.q_to_k_index_dict = self.dataset['q_to_k_index']
        self.length = len(self.k_graphs)
        self.total = self.length
        print(f'=====k_graph_size:{len(self.k_graphs)}, q_graph_size:{len(self.q_graphs)}======')

    def __getitem__(self, idx):
        node_idx = 0
        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.k_graphs[idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()

        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     self.k_graphs[idx],
        #     seeds=[node_idx, other_node_idx],
        #     restart_prob=self.restart_prob,
        #     max_nodes_per_seed=max_nodes_per_seed,
        # )

        graph_q = data_util._my_aug_for_dgl(
            g=self.k_graphs[idx],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._my_ori_dgl(
            g=self.k_graphs[idx],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_k, graph_q


if __name__ == "__main__":
    num_workers = 2
    import psutil

    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    graph_dataset = MyGraphClassificationDataset(
        dataset='mydataset',
        rw_hops=256,
        subgraph_size=300,
        restart_prob=0.8,
        positional_embedding_size=32
    )
    print('')
    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    graph_loader = torch.utils.data.DataLoader(
        graph_dataset,
        batch_size=10,
        collate_fn=data_util.batcher(),
        num_workers=num_workers,
    )
    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    for step, batch in enumerate(graph_loader):
        print("bs", batch[0].batch_size)
        print("n=", batch[0].number_of_nodes())
        print("m=", batch[0].number_of_edges())
        mem = psutil.virtual_memory()
        print(mem.used / 1024 ** 3)
        #  print(batch.graph_q)
        #  print(batch.graph_q.ndata['pos_directed'])
        # print(batch[0].ndata["node_type"])
    exit(0)
    graph_dataset = MyGraphClassificationDataset(
        dataset='mydataset',
        subgraph_size=128,
    )
    graph_loader = torch.utils.data.DataLoader(
        graph_dataset,
        batch_size=10,  # 把几个弄成一组进行返回
        collate_fn=data_util.batcher(),
        num_workers=num_workers,
    )
    for step, batch in enumerate(graph_loader):
        print(batch.graph_q)
        # print(batch.graph_q.ndata["api_pro"].shape)
        print(batch.graph_q.batch_size)
        # print("max", batch.graph_q.edata["efeat"].max())
        break
