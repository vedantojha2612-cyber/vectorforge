"""VectorForge Dataset Synthesizer, High-Level Index Manager & Benchmark Harness."""

from __future__ import annotations
import math
import random
import json
import time
from typing import List, Dict, Any, Optional, Tuple

from .hnsw import HNSWIndex, SearchReceipt
from .distance import project_to_3d


class VectorForge:
    """High-level Vector Search Engine with dataset generation and persistence."""

    def __init__(
        self,
        dim: int = 16,
        metric: str = "euclidean",
        M: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
    ):
        self.dim = dim
        self.metric = metric
        self.index = HNSWIndex(
            dim=dim,
            metric=metric,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )

    def insert(
        self,
        vector: List[float],
        doc_id: Optional[int] = None,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        visual_pos: Optional[Tuple[float, float, float]] = None,
    ) -> int:
        """Adds a single vector document."""
        meta = metadata or {}
        if label:
            meta["label"] = label
        return self.index.insert(
            vector=vector,
            node_id=doc_id,
            metadata=meta,
            visual_pos=visual_pos,
        )

    def search(
        self,
        query: List[float],
        k: int = 5,
        ef_search: Optional[int] = None,
    ) -> SearchReceipt:
        """Executes k-NN search and records complete multi-layer routing trajectory."""
        return self.index.knn_search(query=query, k=k, ef_search=ef_search)

    def seed_synthetic_universe(
        self,
        num_vectors: int = 150,
        num_clusters: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates structured high-dimensional clustered Gaussian data for demos and visualization."""
        rng = random.Random(seed)

        # Cluster centroids positioned in high-dimensional space
        cluster_names = [
            "Quantum-Computing",
            "Neural-Architectures",
            "Cosmic-Astrophysics",
            "Cryptographic-Primes",
            "Distributed-Systems",
            "Synthetic-Genomics",
            "Autonomous-Robotics",
            "Plasma-Dynamics",
        ]

        centroids = []
        for i in range(num_clusters):
            center = [rng.gauss(0.0, 3.0) for _ in range(self.dim)]
            name = cluster_names[i % len(cluster_names)]
            centroids.append((name, center))

        inserted_ids = []
        for idx in range(num_vectors):
            cluster_name, center = rng.choice(centroids)
            # Add Gaussian noise around cluster center
            vec = [c + rng.gauss(0.0, 0.6) for c in center]
            # Visual 3D coordinates based on projected location
            vpos = project_to_3d(vec, seed=seed)

            meta = {
                "cluster": cluster_name,
                "token": f"VEC-{idx:04d}",
                "tag": f"doc_{idx}",
                "magnitude": round(math.sqrt(sum(x * x for x in vec)), 3),
            }
            nid = self.index.insert(vector=vec, node_id=idx, metadata=meta, visual_pos=vpos)
            inserted_ids.append(nid)

        return {
            "total_inserted": len(inserted_ids),
            "clusters": [name for name, _ in centroids],
            "stats": self.index.get_stats(),
        }

    def benchmark_recall(
        self,
        num_queries: int = 20,
        k: int = 5,
        ef_search: Optional[int] = None,
        seed: int = 999,
    ) -> Dict[str, Any]:
        """Evaluates empirical Recall@k, latency, and distance evaluation efficiency."""
        if not self.index.nodes:
            raise ValueError("Index is empty. Insert vectors before benchmarking.")

        rng = random.Random(seed)
        all_nodes = list(self.index.nodes.values())

        total_recalls, = []
        latencies = []
        eval_counts = []
        hops = []

        for _ in range(num_queries):
            # Pick a sample vector with slight perturbation or random probe
            base = rng.choice(all_nodes).vector
            probe = [x + rng.gauss(0.0, 0.2) for x in base]

            # Ground truth exact brute force
            exact_top = self.index.brute_force_knn(probe, k=k)
            exact_ids = {nid for _, nid in exact_top}

            # HNSW approximate search
            receipt = self.index.knn_search(probe, k=k, ef_search=ef_search)
            hnsw_ids = {item["id"] for item in receipt.top_k}

            intersection = exact_ids.intersection(hnsw_ids)
            recall = len(intersection) / float(k)
            total_recalls.append(recall)
            latencies.append(receipt.latency_ms)
            eval_counts.append(receipt.distance_evaluations)
            hops.append(receipt.hop_count)

        avg_recall = sum(total_recalls) / len(total_recalls)
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        avg_evals = sum(eval_counts) / len(eval_counts)
        avg_hops = sum(hops) / len(hops)

        # Efficiency ratio: distance evals vs linear brute force
        total_items = len(self.index.nodes)
        speedup = total_items / max(1.0, avg_evals)

        return {
            "num_queries": num_queries,
            "k": k,
            "ef_search": ef_search or self.index.ef_search,
            "mean_recall_at_k": round(avg_recall, 4),
            "avg_latency_ms": round(avg_latency, 3),
            "p95_latency_ms": round(p95_latency, 3),
            "avg_distance_evaluations": round(avg_evals, 1),
            "avg_hops": round(avg_hops, 1),
            "theoretical_speedup": round(speedup, 2),
            "total_nodes": total_items,
        }

    def save_to_file(self, filepath: str) -> None:
        """Persists graph index state and metadata to JSON file."""
        stats = self.index.get_stats()
        nodes_dump = []
        for nid, node in self.index.nodes.items():
            nodes_dump.append({
                "id": node.id,
                "vector": node.vector,
                "level": node.level,
                "metadata": node.metadata,
                "visual_pos": node.visual_pos,
            })

        layers_dump = []
        for layer in self.index.layers:
            layers_dump.append({str(k): list(v) for k, v in layer.items()})

        data = {
            "dim": self.dim,
            "metric": self.metric,
            "M": self.index.M,
            "M0": self.index.M0,
            "ef_construction": self.index.ef_construction,
            "ef_search": self.index.ef_search,
            "entry_point": self.index.entry_point,
            "max_level": self.index.max_level,
            "nodes": nodes_dump,
            "layers": layers_dump,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> VectorForge:
        """Loads persisted graph index from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vf = cls(
            dim=data["dim"],
            metric=data["metric"],
            M=data["M"],
            ef_construction=data["ef_construction"],
            ef_search=data["ef_search"],
        )
        vf.index.entry_point = data["entry_point"]
        vf.index.max_level = data["max_level"]

        from .hnsw import Node
        for nd in data["nodes"]:
            node = Node(
                id=nd["id"],
                vector=nd["vector"],
                level=nd["level"],
                metadata=nd.get("metadata", {}),
                visual_pos=tuple(nd.get("visual_pos", (0.0, 0.0, 0.0))),
            )
            vf.index.nodes[node.id] = node

        vf.index.layers = []
        for ld in data["layers"]:
            layer_dict = {int(k): set(v) for k, v in ld.items()}
            vf.index.layers.append(layer_dict)

        return vf
