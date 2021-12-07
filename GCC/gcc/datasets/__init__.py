from .graph_dataset import (
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)

from .my_graph_dataset import (
    MyGraphClassificationDataset,
    MyGraphClassificationDatasetForOnlyOrigin,
)
GRAPH_CLASSIFICATION_DSETS = ["collab", "imdb-binary", "imdb-multi", "rdt-b", "rdt-5k"]

__all__ = [
    "GRAPH_CLASSIFICATION_DSETS",
    "LoadBalanceGraphDataset",
    "GraphClassificationDataset",
    "MyGraphClassificationDataset",
    "GraphClassificationDatasetLabeled",
    "NodeClassificationDataset",
    "NodeClassificationDatasetLabeled",
    "MyGraphClassificationDatasetForOnlyOrigin",
    "worker_init_fn",
]

