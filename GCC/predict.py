#!/usr/bin/env python
# encoding: utf-8
# File Name: predict.py
# Author: Zhen Ziyang
# Create Time: 2022/04/28 16:44
# TODO:

import argparse
import os
import pickle
import time

import dgl
import numpy as np
# import tensorboard_logger as tb_logger
import torch
import warnings
import sys
from gcc.Sample import Sample, Node
from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
    MyGraphClassificationDataset,
)
from gcc.datasets.data_util import batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear

warnings.filterwarnings("ignore")
root_path = os.path.abspath("./")
sys.path.append(root_path)
from gcc.Sample import Sample, Node
from gcc.datasets.data_util import create_graph_classification_dataset
from gcc.tasks import build_model


def generate_emb(graphs, model):
    model.eval()
    emb_list = []
    for graph in graphs:

        with torch.no_grad():
            feat = model(graph)

        emb_list.append(feat.detach().cpu())
    return torch.cat(emb_list)


def do_generate_emb(args_test):
    if os.path.isfile(args_test.load_path):
        print("=> loading checkpoint '{}'".format(args_test.load_path))
        checkpoint = torch.load(args_test.load_path, map_location="cpu")
        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args_test.load_path, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args_test.load_path))
    args = checkpoint["opt"]

    # assert args_test.gpu is None or torch.cuda.is_available()
    print("Use GPU: {} for generation".format(args_test.gpu))
    args.gpu = args_test.gpu
    args.device = torch.device("cpu") if args.gpu is None else torch.device(args.gpu)

    if args_test.dataset in GRAPH_CLASSIFICATION_DSETS or args_test.dataset == 'mydataset':
        train_dataset = MyGraphClassificationDataset(
            dataset=args_test.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    else:
        train_dataset = NodeClassificationDataset(
            dataset=args_test.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    args.batch_size = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # create model and optimizer
    model = GraphEncoder(
        positional_embedding_size=args.positional_embedding_size,
        max_node_freq=args.max_node_freq,
        max_edge_freq=args.max_edge_freq,
        max_degree=args.max_degree,
        freq_embedding_size=args.freq_embedding_size,
        degree_embedding_size=args.degree_embedding_size,
        output_dim=args.hidden_size,
        node_hidden_dim=args.hidden_size,
        edge_hidden_dim=args.hidden_size,
        num_layers=args.num_layer,
        num_step_set2set=args.set2set_iter,
        num_layer_set2set=args.set2set_lstm_layer,
        gnn_model=args.model,
        norm=args.norm,
        degree_input=True,
    )

    model = model.to(torch.device(args_test.gpu if torch.cuda.is_available() else "cpu"))

    model.load_state_dict(checkpoint["model"])

    del checkpoint

    emb = generate_emb(train_loader, model, args)
    return emb


def greate_emb():
    parser = argparse.ArgumentParser("argument for training")
    # fmt: off
    parser.add_argument("--load-path", type=str, help="path to load model")
    parser.add_argument("--dataset", type=str, default="dgl",
                        choices=["mydataset", "dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport",
                                 "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm",
                                 "sigmod", "icde", "h-index-rand-1", "h-index-top-1",
                                 "h-index"] + GRAPH_CLASSIFICATION_DSETS)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    # fmt: on
    do_generate_emb(parser.parse_args())


GRAPH_OUTPUT_PATH = 'gcc/result/'

family_model_name = "./gcc/result/model_family.pickle.dat"
big_label_model_name = "./gcc/result/model_big_label.pickle.dat"


def k_label_to_q_label(label_k, q_to_k_index):
    label_q = []
    for k_index in q_to_k_index:
        label_q.append(label_k[k_index])
    return label_q


class GraphClassification(object):
    def __init__(self, dataset, model, hidden_size, num_shuffle, seed, **model_args):
        assert model == "from_numpy_graph"
        dataset = create_graph_classification_dataset()
        self.num_nodes = len(dataset['graph_labels'])
        self.num_classes = dataset['num_labels']
        self.label_matrix = np.zeros((self.num_nodes, self.num_classes), dtype=int)
        self.labels = np.array(k_label_to_q_label(dataset['graph_labels'], dataset['q_to_k_index']))
        # self.labels = np.array(dataset.graph_labels)
        self.big_labels = np.array(k_label_to_q_label(dataset['graph_big_labels'], dataset['q_to_k_index']))
        self.model = build_model(model, hidden_size, **model_args)
        self.hidden_size = hidden_size
        self.num_shuffle = num_shuffle
        self.seed = seed
        self.test_graphs = []
        print(f'self labels')
        print(self.labels)
        print(f'self big labels')
        print(self.big_labels[0:20])

    def predict(self, embeddings):
        # TODO self.test_graphs转embeddings
        result = {}
        result['family_result'] = self.svc_predict(embeddings, family_model_name)
        result['big_label_result'] = self.svc_predict(embeddings, big_label_model_name)
        return result

    def svc_predict(self, x, model_path):
        loaded_model = pickle.load(open(model_path, "rb"))
        y_pred = loaded_model.predict(x)
        print(f'y_pred: {y_pred}')
        return y_pred


def do_predict(emb):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shuffle", type=int, default=10)
    parser.add_argument("--emb-path", type=str, default="")
    args = parser.parse_args()
    task = GraphClassification(
        args.dataset,
        args.model,
        args.hidden_size,
        args.num_shuffle,
        args.seed,
        emb_path=args.emb_path,
    )
    ret = task.predict(emb)
    print(ret)


if __name__ == "__main__":
    graphs = []
    emb = greate_emb(graphs)
    do_predict(emb)
