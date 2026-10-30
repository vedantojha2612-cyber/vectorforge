"""VectorForge REST API Server in Pure Python stdlib.

Provides HTTP JSON endpoints for graph topology streaming, real-time vector queries,
trajectory telemetry for WebGL 3D holographic rendering, and benchmark execution.
"""

from __future__ import annotations
import http.server
import json
import os
import random
import sys
import urllib.parse
from typing import Optional, Dict, Any

from .index import VectorForge


class VectorForgeHandler(http.server.BaseHTTPRequestHandler):
    """Zero-dependency HTTP request handler for VectorForge REST & WebGL HUD."""

    engine: VectorForge = None  # Class-level engine instance
    web_dir: Optional[str] = None

    def _set_security_headers(self, content_type: str = "application/json"):
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")

    def do_OPTIONS(self):
        """Handle CORS pre-flight."""
        self.send_response(204)
        self._set_security_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests for static assets and API state."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/stats":
            self._send_json(self.engine.index.get_stats())
            return

        if path == "/api/topology":
            self._send_json(self.engine.index.export_graph_topology())
            return

        if path == "/api/benchmark":
            query_params = urllib.parse.parse_qs(parsed_url.query)
            n_queries = int(query_params.get("n", [15])[0])
            k = int(query_params.get("k", [5])[0])
            ef = int(query_params.get("ef", [32])[0])
            try:
                results = self.engine.benchmark_recall(num_queries=n_queries, k=k, ef_search=ef)
                self._send_json(results)
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        # Serve static web directory if configured
        if self.web_dir and os.path.exists(self.web_dir):
            if path in ("/", ""):
                file_path = os.path.join(self.web_dir, "index.html")
            else:
                rel = path.lstrip("/")
                file_path = os.path.realpath(os.path.join(self.web_dir, rel))
                # Path traversal defense
                if not file_path.startswith(os.path.realpath(self.web_dir)):
                    self.send_response(403)
                    self._set_security_headers("text/plain")
                    self.end_headers()
                    self.wfile.write(b"403 Forbidden")
                    return

            if os.path.exists(file_path) and os.path.isfile(file_path):
                self._serve_file(file_path)
                return

        # Fallback API root
        self._send_json({
            "service": "VectorForge HNSW Engine",
            "version": "0.1.0",
            "status": "online",
            "endpoints": [
                "/api/stats",
                "/api/topology",
                "/api/search",
                "/api/insert",
                "/api/seed",
                "/api/benchmark",
            ],
            "stats": self.engine.index.get_stats(),
        })

    def do_POST(self):
        """Handle POST requests for vector search, insertions, and seeds."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            self._send_json({"error": f"Invalid JSON payload: {e}"}, status=400)
            return

        if path == "/api/search":
            try:
                k = int(payload.get("k", 5))
                ef = int(payload.get("ef_search", 32))
                query_vec = payload.get("query")

                if query_vec is None:
                    # Pick a probe from existing nodes or random vector
                    if self.engine.index.nodes:
                        sample_node = random.choice(list(self.engine.index.nodes.values()))
                        query_vec = [x + random.gauss(0.0, 0.1) for x in sample_node.vector]
                    else:
                        query_vec = [random.gauss(0.0, 1.0) for _ in range(self.engine.dim)]

                receipt = self.engine.search(query=query_vec, k=k, ef_search=ef)
                self._send_json({
                    "query": receipt.query,
                    "top_k": receipt.top_k,
                    "trajectory": receipt.trajectory,
                    "distance_evaluations": receipt.distance_evaluations,
                    "hop_count": receipt.hop_count,
                    "entry_point": receipt.entry_point,
                    "entry_level": receipt.entry_level,
                    "latency_ms": receipt.latency_ms,
                    "ef_search": receipt.ef_search,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/insert":
            try:
                vec = payload.get("vector")
                if not vec or len(vec) != self.engine.dim:
                    self._send_json({"error": f"Vector must have dimension {self.engine.dim}"}, status=400)
                    return
                doc_id = payload.get("id")
                label = payload.get("label", "")
                metadata = payload.get("metadata", {})
                nid = self.engine.insert(vector=vec, doc_id=doc_id, label=label, metadata=metadata)
                self._send_json({"status": "inserted", "node_id": nid, "stats": self.engine.index.get_stats()})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path = = "/api/seed":
            try:
                count = int(payload.get("count", 150))
                clusters = int(payload.get("clusters", 5))
                # Reinitialize clean index and seed
                self.engine.index = type(self.engine.index)(
                    dim=self.engine.dim,
                    metric=self.engine.metric,
                    M=self.engine.index.M,
                    ef_construction=self.engine.index.ef_construction,
                    ef_search=self.engine.index.ef_search,
                )
                res = self.engine.seed_synthetic_universe(num_vectors=count, num_clusters=clusters)
                self._send_json({"status": "seeded", "result": res})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": f"Endpoint '{path}' not found"}, status=404)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self._set_security_headers("application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }
        ctype = mime_types.get(ext, "application/octet-stream")
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self._set_security_headers(ctype)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"500 Internal Error: {e}".encode("utf-8"))

    def log_message(self, format: str, *args: Any):
        """Suppresses default stderr logging unless debugging."""
        pass


def run_server(
    engine: VectorForge,
    host: str = "127.0.0.1",
    port: int = 8765,
    web_dir: Optional[str] = None,
):
    """Starts the VectorForge HTTP server."""
    VectorForgeHandler.engine = engine
    VectorForgeHandler.web_dir = web_dir

    server = http.server.ThreadingHTTPServer((host, port), VectorForgeHandler)
    print(f"⚡ [VectorForge] REST Engine & Hologram Stream active on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⚡ [VectorForge] Server gracefully stopped.")
    finally:
        server.server_close()
