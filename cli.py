#!/usr/bin/env python3
"""VectorForge Unified CLI & Systems Command Interface.

Pure Python stdlib CLI for HNSW vector graph indexing, routing trajectory inspection,
recall@k benchmarking, 15 FPS ANSI/Braille Terminal Radar HUD, and holographic telemetry streaming.
"""

from __future__ import annotations
import argparse
import os
import sys
import time

from vectorforge.index import VectorForge
from vectorforge.server import run_server
from vectorforge.tui import run_terminal_radar


def print_banner():
    banner = """
\033[1;36m╦  ╦┌─┐┌─┐┌┬┐┌─┐┬─┐╔═╗┌─┐┬─┐┌─┐┌─┐
\033[1;36m║  ║├┤ │   │ │ │├┬┘╠═╣│ │├┬┘│ ┬├┤ 
\033[1;34m╩═╝└─┘└─┘ ┴ └─┘┴└─╩ ╩└─┘┴└─└─┘└─┘\033[0m
\033[1;30mReal-Time 3D HNSW High-Dimensional Vector Search Engine (Pure Python Stdlib)\033[0m
"""
    print(banner)


def cmd_demo(args):
    print_banner()
    print(f"\033[1;33m[*] Initializing VectorForge HNSW Engine (dim={args.dim}, M={args.M}, efConstruction={args.ef_construction})...\033[0m")
    vf = VectorForge(
        dim=args.dim,
        metric=args.metric,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    print(f"[*] Seeding synthetic high-dimensional universe ({args.count} vectors across {args.clusters} clusters)...")
    res = vf.seed_synthetic_universe(num_vectors=args.count, num_clusters=args.clusters)
    stats = vf.index.get_stats()
    print(f"\033[1;32m[+] Graph Built:\033[0m {stats['total_nodes']} nodes, {stats['total_edges']} edges, Max Layer: {stats['max_level']}")
    print(f"    Layer distribution: {stats['layer_distribution']}")

    print(f"\n\033[1;35m[*] Executing Sample k-NN Trajectory Routing (k={args.k}, efSearch={args.ef_search})...\033[0m")
    # Sample probe from universe with small perturbation
    sample_node = vf.index.nodes[0]
    probe = [x + 0.1 for x in sample_node.vector]

    receipt = vf.search(query=probe, k=args.k, ef_search=args.ef_search)
    print(f"\033[1;32m[+] Search Completed in {receipt.latency_ms:.3f} ms!\033[0m")
    print(f"    Distance Evaluations: {receipt.distance_evaluations} | Hops: {receipt.hop_count}")
    print(f"    Entry Point: Node #{receipt.entry_point} at Layer {receipt.entry_level}")

    print("\n\033[1;36m── Routing Trajectory Trace ──────────────────────────────────────────\033[0m")
    for step in receipt.trajectory:
        act = step["action"]
        lyr = step["layer"]
        fn = f"#{step['from_node']}" if step['from_node'] is not None else "START"
        tn = f"#{step['to_node']}"
        dst = step["distance"]
        color = "\033[1;33m" if act == "greedy_hop" else ("\033[1;35m" if act == "layer_descent" else "\033[1;34m")
        print(f"  {color}[L{lyr:02d}] {act:<15}\033[0m {fn:>6} ➔ {tn:<6} | dist: {dst:.4f}")

    print("\n\033[1;32m── Top-k Nearest Neighbors ───────────────────────────────────────────\033[0m")
    for rank, item in enumerate(receipt.top_k, 1):
        meta = item["metadata"]
        cluster = meta.get("cluster", "Unknown")
        tok = meta.get("token", "")
        print(f"  \033[1m#{rank}\033[0m Node #{item['id']:<4} | Dist: {item['distance']:.4f} | Sim: {item['similarity']:.4f} | Cluster: {cluster:<22} | {tok}")


def cmd_benchmark(args):
    print_banner()
    print(f"\033[1;33m[*] VectorForge Systems Benchmark: Recall@k & Latency Profiling\033[0m")
    print(f"    Index Size: {args.count} vectors (dim={args.dim}, M={args.M}) | Metric: {args.metric}")
    print(f"    Ground Truth: Exact Linear Scan vs HNSW Greedy Traversal\n")

    vf = VectorForge(
        dim=args.dim,
        metric=args.metric,
        M=args.M,
        ef_construction=args.ef_construction,
    )
    t0 = time.perf_counter()
    vf.seed_synthetic_universe(num_vectors=args.count, num_clusters=args.clusters)
    build_time = (time.perf_counter() - t0) * 1000.0
    print(f"\033[1;32m[+] Index Construction Complete:\033[0m {build_time:.1f} ms\n")

    ef_candidates = [8, 16, 32, 64, 128]
    print("┌──────┬──────────┬───────────┬─────────────┬───────────┬──────────────┐")
    print("│  k   │ efSearch │ Recall@k  │ Avg Latency │ P95 Lat   │ Evals / Hop  │")
    print("├──────┼──────────┼───────────┼─────────────┼───────────┼──────────────┤")

    for ef in ef_candidates:
        res = vf.benchmark_recall(num_queries=args.queries, k=args.k, ef_search=ef)
        recall_pct = f"{res['mean_recall_at_k'] * 100.0:.1f}%"
        avg_lat = f"{res['avg_latency_ms']:.2f} ms"
        p95_lat = f"{res['p95_latency_ms']:.2f} ms"
        evals_str = f"{res['avg_distance_evaluations']:.0f} / {res['avg_hops']:.0f}"
        print(f"│ {args.k:>4} │ {ef:>8} │ {recall_pct:>9} │ {avg_lat:>11} │ {p95_lat:>9} │ {evals_str:>12} │")

    print("└──────┴──────────┴───────────┴─────────────┴───────────┴──────────────┘")
    stats = vf.index.get_stats()
    print(f"\n[+] Total Nodes: {stats['total_nodes']} | Max Layer: {stats['max_level']} | Layers: {stats['layer_distribution']}")


def cmd_serve(args):
    print_banner()
    vf = VectorForge(
        dim=args.dim,
        metric=args.metric,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    print(f"[*] Pre-seeding universe with {args.count} nodes across {args.clusters} clusters...")
    vf.seed_synthetic_universe(num_vectors=args.count, num_clusters=args.clusters)

    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
    run_server(engine=vf, host=args.host, port=args.port, web_dir=web_dir)


def cmd_tui(args):
    print_banner()
    vf = VectorForge(
        dim=args.dim,
        metric=args.metric,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    print(f"[*] Initializing Terminal Radar with {args.count} nodes across {args.clusters} clusters...")
    vf.seed_synthetic_universe(num_vectors=args.count, num_clusters=args.clusters)
    run_terminal_radar(vf, target_fps=15.0)


def main():
    parser = argparse.ArgumentParser(
        description="VectorForge: Real-Time 3D HNSW High-Dimensional Vector Search Engine (Pure Python stdlib)",
    )
    parser.add_argument("--web", action="store_true", help="Launch 3D WebGL Hologram telemetry server")
    parser.add_argument("--tui", action="store_true", help="Launch 15 FPS ANSI/Braille Terminal Radar HUD")

    subparsers = parser.add_subparsers(dest="command")

    # Demo
    p_demo = subparsers.add_parser("demo", help="Run interactive HNSW search & trajectory trace demo")
    p_demo.add_argument("--dim", type=int, default=16, help="Vector dimension (default: 16)")
    p_demo.add_argument("--count", type=int, default=150, help="Number of vectors (default: 150)")
    p_demo.add_argument("--clusters", type=int, default=5, help="Number of clusters (default: 5)")
    p_demo.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine", "dot"])
    p_demo.add_argument("-M", type=int, default=16, help="Max outgoing links per node (default: 16)")
    p_demo.add_argument("--ef-construction", type=int, default=64, help="efConstruction beam size")
    p_demo.add_argument("--ef-search", type=int, default=32, help="efSearch beam size")
    p_demo.add_argument("-k", type=int, default=5, help="Number of nearest neighbors to retrieve")

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run Recall@k vs QPS throughput benchmark")
    p_bench.add_argument("--dim", type=int, default=16, help="Vector dimension (default: 16)")
    p_bench.add_argument("--count", type=int, default=200, help="Number of vectors (default: 200)")
    p_bench.add_argument("--clusters", type=int, default=5, help="Number of clusters (default: 5)")
    p_bench.add_argument("--queries", type=int, default=30, help="Number of probe queries (default: 30)")
    p_bench.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine", "dot"])
    p_bench.add_argument("-M", type=int, default=16, help="Max outgoing links per node (default: 16)")
    p_bench.add_argument("--ef-construction", type=int, default=64, help="efConstruction beam size")
    p_bench.add_argument("-k", type=int, default=5, help="Number of nearest neighbors to retrieve")

    # Serve
    p_serve = subparsers.add_parser("serve", help="Run REST API & WebGL Hologram telemetry server")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    p_serve.add_argument("--dim", type=int, default=16, help="Vector dimension (default: 16)")
    p_serve.add_argument("--count", type=int, default=150, help="Initial vectors (default: 150)")
    p_serve.add_argument("--clusters", type=int, default=5, help="Clusters count (default: 5)")
    p_serve.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine", "dot"])
    p_serve.add_argument("-M", type=int, default=16, help="Max outgoing links per node (default: 16)")
    p_serve.add_argument("--ef-construction", type=int, default=64, help="efConstruction beam size")
    p_serve.add_argument("--ef-search", type=int, default=32, help="efSearch beam size")

    # TUI
    p_tui = subparsers.add_parser("tui", help="Run 15 FPS ANSI/Braille Terminal Radar HUD")
    p_tui.add_argument("--dim", type=int, default=16, help="Vector dimension (default: 16)")
    p_tui.add_argument("--count", type=int, default=150, help="Number of vectors (default: 150)")
    p_tui.add_argument("--clusters", type=int, default=5, help="Clusters count (default: 5)")
    p_tui.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine", "dot"])
    p_tui.add_argument("-M", type=int, default=16, help="Max outgoing links per node (default: 16)")
    p_tui.add_argument("--ef-construction", type=int, default=64, help="efConstruction beam size")
    p_tui.add_argument("--ef-search", type=int, default=32, help="efSearch beam size")

    args = parser.parse_args()

    # Route based on flags or subcommand
    if args.web:
        args.host = "127.0.0.1"
        args.port = 8765
        args.dim = 16
        args.count = 150
        args.clusters = 5
        args.metric = "euclidean"
        args.M = 16
        args.ef_construction = 64
        args.ef_search = 32
        cmd_serve(args)
    elif args.tui or args.command == "tui":
        if not hasattr(args, "dim"):
            args.dim = 16
            args.count = 150
            args.clusters = 5
            args.metric = "euclidean"
            args.M = 16
            args.ef_construction = 64
            args.ef_search = 32
        cmd_tui(args)
    elif not args.command or args.command == "demo":
        if not hasattr(args, "dim"):
            args.dim = 16
            args.count = 150
            args.clusters = 5
            args.metric = "euclidean"
            args.M = 16
            args.ef_construction = 64
            args.ef_search = 32
            args.k = 5
        cmd_demo(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
