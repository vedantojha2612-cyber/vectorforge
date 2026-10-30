"""Unit tests for VectorForge REST API server endpoints."""

import json
import threading
import time
import unittest
import urllib.request
from vectorforge.index import VectorForge
from vectorforge.server import run_server


class TestVectorForgeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = VectorForge(dim=6, metric="euclidean", M=8, ef_construction=24, ef_search=16)
        cls.engine.seed_synthetic_universe(num_vectors=40, num_clusters=3, seed=42)
        cls.port = 8991
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        # Start server in daemon thread
        cls.server_thread = threading.Thread(
            target=run_server,
            kwargs={"engine": cls.engine, "host": "127.0.0.1", "port": cls.port},
            daemon=True,
        )
        cls.server_thread.start()
        time.sleep(0.3)

    def test_1_stats_endpoint(self):
        req = urllib.request.Request(f"{self.base_url}/api/stats")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertGreaterEqual(data["total_nodes"], 40)
            self.assertIn("layer_distribution", data)

    def test_2_topology_endpoint(self):
        req = urllib.request.Request(f"{self.base_url}/api/topology")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("nodes", data)
            self.assertIn("links", data)
            self.assertGreaterEqual(len(data["nodes"]), 40)

    def test_3_search_endpoint(self):
        payload = json.dumps({"k": 3, "ef_search": 16}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(len(data["top_k"]), 3)
            self.assertIn("trajectory", data)
            self.assertGreater(data["distance_evaluations"], 0)

    def test_4_insert_endpoint(self):
        curr_total = len(self.engine.index.nodes)
        payload = json.dumps({
            "vector": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "label": "Manual-Insert",
            "metadata": {"test": True},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/insert",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["status"], "inserted")
            self.assertEqual(len(self.engine.index.nodes), curr_total + 1)


if __name__ == "__main__":
    unittest.main()
