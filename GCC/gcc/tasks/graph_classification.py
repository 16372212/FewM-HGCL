import argparse
import copy
import random
import io
import warnings
from collections import defaultdict
import os, sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

# 为了让gcc添加到源路径
warnings.filterwarnings("ignore")
root_path = os.path.abspath("./")
sys.path.append(root_path)
from gcc.Sample import Sample, Node
from gcc.datasets.data_util import create_graph_classification_dataset
from gcc.tasks import build_model

warnings.filterwarnings("ignore")


GRAPH_OUTPUT_PATH = 'gcc/result/'


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
        print(f'self labels')
        print(self.labels)
        print(f'self big labels')
        print(self.big_labels[0:20])


    def train(self):
        embeddings = self.model.train(None)
        result = {}
        result['family_result'] = self.svc_classify(embeddings, self.labels, False)
        result['big_label_result'] = self.svc_classify(embeddings, self.big_labels, False)
        return result


    def svc_classify(self, x, y, search):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        accuracies = []
        recall = []
        precision = []
        f1_scores = []
        for train_index, test_index in kf.split(x, y):

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
            if search:
                params = {"C": [1, 10, 100, 1000, 10000, 100000]}
                classifier = GridSearchCV(
                    SVC(), params, cv=5, scoring="accuracy", verbose=0, n_jobs=-1
                )
            else:
                classifier = SVC(C=100000)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            print('y_pred:')
            print(y_pred)
            print('y_test:')
            print(y_test)
            print(f'acc: {accuracy_score(y_test, y_pred)}')
            recall.append(recall_score(y_test, y_pred, average='weighted'))
            precision.append(precision_score(y_test, y_pred, average='weighted'))
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            break
        print(f'f1-score: {f1_scores}')
        print(f'Micro-F1: {np.mean(accuracies)},precision:{np.mean(precision)},"recall":{np.mean(recall)},"f1-score":{np.mean(f1_scores)}')
        return {"acc": np.mean(accuracies),"precision":np.mean(precision),"recall":np.mean(recall),"f1-score":np.mean(f1_scores)}


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
    ret = task.train()
    print(ret)
    # write result
    f1 = open(args.emb_path.split('/')[0]+'/result.txt','w')
    f1.write(str(ret))
    f1.close()
