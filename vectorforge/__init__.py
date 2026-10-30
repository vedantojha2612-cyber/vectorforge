"""VectorForge: Real-Time 3D HNSW High-Dimensional Vector Search Engine in Pure Python."""

from .distance import (
    euclidean_distance,
    cosine_distance,
    dot_product,
    normalize,
    project_to_3d,
)
from .hnsw import HNSWIndex, SearchReceipt, Node
from .index import VectorForge

__version__ = "0.1.0"
__all__ = [
    "HNSWIndex",
    "VectorForge",
    "SearchReceipt",
    "Node",
    "euclidean_distance",
    "cosine_distance",
    "dot_product",
    "normalize",
    "project_to_3d",
]
