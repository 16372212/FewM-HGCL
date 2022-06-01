import argparse
import copy
import pickle
import random
import io
import warnings
from collections import defaultdict
import os, sys
import networkx as nx
import numpy as np

# 为了让gcc添加到源路径
warnings.filterwarnings("ignore")
root_path = os.path.abspath("./")
sys.path.append(root_path)
from gcc.Sample import Sample, Node
from gcc.datasets.data_util import create_graph_classification_dataset
from gcc.tasks import build_model


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

    def predict(self):
        # TODO self.test_graphs转embeddings
        embeddings = [[]]
        result = {}
        result['family_result'] = self.svc_predict(embeddings, family_model_name)
        result['big_label_result'] = self.svc_predict(embeddings, big_label_model_name)
        return result

    def svc_predict(self, x, model_path):
        loaded_model = pickle.load(open(model_path, "rb"))
        y_pred = loaded_model.predict(x)
        print(f'y_pred: {y_pred}')
        return y_pred


if __name__ == "__main__":
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
    ret = task.predict()
    print(ret)
    # write result
