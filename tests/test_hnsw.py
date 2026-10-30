"""Unit tests for HNSW graph structure, convergence, and trajectory telemetry."""

import unittest
from vectorforge.hnsw import HNSWIndex
from vectorforge.index import VectorForge


class TestHNSWIndex(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.index = HNSWIndex(dim=self.dim, metric="euclidean", M=8, ef_construction=32, ef_search=16, seed=123)

    def test_single_insert_and_search(self):
        vec = [1.0] * self.dim
        nid = self.index.insert(vec, node_id=0, metadata={"name": "first"})
        self.assertEqual(nid, 0)
        self.assertEqual(self.index.entry_point, 0)

        receipt = self.index.knn_search(vec, k=1)
        self.assertEqual(len(receipt.top_k), 1)
        self.assertEqual(receipt.top_k[0]["id"], 0)
        self.assertAlmostEqual(receipt.top_k[0]["distance"], 0.0, places=5)

    def test_multi_insert_and_recall(self):
        # Insert 50 vectors
        vf = VectorForge(dim=8, metric="euclidean", M=12, ef_construction=48, ef_search=24)
        vf.seed_synthetic_universe(num_vectors=60, num_clusters=4, seed=42)

        stats = vf.index.get_stats()
        self.assertEqual(stats["total_nodes"], 60)
        self.assertGreater(stats["max_level"], 0)
        self.assertGreater(stats["total_edges"], 0)

        # Benchmark recall
        bench = vf.benchmark_recall(num_queries=15, k=5, ef_search=32)
        # Recall@5 with ef_search=32 should be high (> 0.85)
        self.assertGreaterEqual(bench["mean_recall_at_k"], 0.80)
        self.assertGreater(bench["avg_distance_evaluations"], 0)
        self.assertGreater(bench["avg_hops"], 0)

    def test_trajectory_recording(self):
        vf = VectorForge(dim=6, metric="euclidean", M=8, ef_construction=32, ef_search=16)
        vf.seed_synthetic_universe(num_vectors=30, num_clusters=3, seed=77)

        sample = vf.index.nodes[0].vector
        receipt = vf.search(sample, k=3)

        self.assertIsNotNone(receipt.entry_point)
        self.assertGreater(len(receipt.trajectory), 0)

        # Verify trajectory step keys
        for step in receipt.trajectory:
            self.assertIn("layer", step)
            self.assertIn("to_node", step)
            self.assertIn("distance", step)
            self.assertIn("action", step)
            self.assertIn("visual_pos", step)

    def test_degree_limits(self):
        M = 8
        M0 = 2 * M
        index = HNSWIndex(dim=4, metric="euclidean", M=M, ef_construction=32, seed=99)
        # Insert 40 random points
        import random
        rng = random.Random(99)
        for i in range(40):
            v = [rng.random() for _ in range(4)]
            index.insert(v, node_id=i)

        # Layer 0 maximum connections must not exceed M0
        for nid, neighbors in index.layers[0].items():
            self.assertLessEqual(len(neighbors), M0)

        # Layers > 0 maximum connections must not exceed M
        for l in range(1, len(index.layers)):
            for nid, neighbors in index.layers[l].items():
                self.assertLessEqual(len(neighbors), M)

    def test_serialization(self):
        import tempfile
        import os

        vf = VectorForge(dim=4, metric="euclidean", M=6, ef_construction=16)
        vf.seed_synthetic_universe(num_vectors=25, num_clusters=3, seed=12)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            vf.save_to_file(tmp_path)
            loaded = VectorForge.load_from_file(tmp_path)

            self.assertEqual(len(loaded.index.nodes), len(vf.index.nodes))
            self.assertEqual(loaded.index.entry_point, vf.index.entry_point)
            self.assertEqual(loaded.index.max_level, vf.index.max_level)

            # Test query on loaded index
            receipt1 = vf.search([0.5, 0.5, 0.5, 0.5], k=3)
            receipt2 = loaded.search([0.5, 0.5, 0.5, 0.5], k=3)
            self.assertEqual([x["id"] for x in receipt1.top_k], [x["id"] for x in receipt2.top_k])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
