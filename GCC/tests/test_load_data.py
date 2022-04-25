import argparse
import os
import dgl

from gcc.tasks.graph_classification import GraphClassification
from tests.utils import E2E_PATH, MOCO_PATH, get_default_args, generate_emb

GRAPH_INPUT_PATH = '../mid_data/gcc_input/subgraphs_train_data.bin'


def test_load_graphs():

    graph_k_list, label_lists = dgl.data.utils.load_graphs(GRAPH_INPUT_PATH)