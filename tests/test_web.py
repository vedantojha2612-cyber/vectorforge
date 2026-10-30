"""Unit tests for VectorForge WebGL Static Asset Serving and Security Headers."""

import os
import threading
import time
import unittest
import urllib.request
import urllib.error
from vectorforge.index import VectorForge
from vectorforge.server import run_server


class TestVectorForgeWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = VectorForge(dim=4, metric="euclidean", M=4, ef_construction=16, ef_search=8)
        cls.engine.seed_synthetic_universe(num_vectors=10, num_clusters=2, seed=42)
        cls.port = 8993
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

        cls.server_thread = threading.Thread(
            target=run_server,
            kwargs={
                "engine": cls.engine,
                "host": "127.0.0.1",
                "port": cls.port,
                "web_dir": cls.web_dir,
            },
            daemon=True,
        )
        cls.server_thread.start()
        time.sleep(0.3)

    def test_static_index_serving(self):
        req = urllib.request.Request(f"{self.base_url}/")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
            self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
            body = resp.read().decode("utf-8")
            self.assertIn("VectorForge", body)
            self.assertIn("3D Multi-Tier HNSW", body)

    def test_path_traversal_blocked(self):
        # Attempt path traversal
        req = urllib.request.Request(f"{self.base_url}/../../cli.py")
        try:
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 403)
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 403)


if __name__ == "__main__":
     unittest.main()
