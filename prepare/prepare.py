from draw_graph import create_graph_matrix
from Sample import Node, Sample
from matrix_to_huge_dgl import draw_aug_dgls, draw_dgl_from_matrix


if __name__ == "__main__":

    # 1. create graph matrix. Read data from mongoDB, use some of them to create
    create_graph_matrix()
    # 2. create dgl
    huge_graph, sample_id_lists, family_label_lists, big_label_lists = draw_dgl_from_matrix()
    # 3. Data augmentation. creating augmented graphs
    draw_aug_dgls(huge_graph, sample_id_lists, family_label_lists, big_label_lists)
