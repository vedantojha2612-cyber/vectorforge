"""VectorForge 15 FPS ANSI/Braille Terminal Radar HUD.

Zero-dependency pure Python stdlib real-time cyberpunk terminal dashboard
for HNSW high-dimensional vector search, spatial projection radar,
and trajectory telemetry.
"""

from __future__ import annotations
import math
import os
import random
import re
import select
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .index import VectorForge
from .hnsw import SearchReceipt


# ANSI Escape Codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

# Cyberpunk Color Palette
CYAN = "\033[38;5;51m"
BRIGHT_CYAN = "\033[38;5;87m"
GOLD = "\033[38;5;220m"
AMBER = "\033[38;5;214m"
EMERALD = "\033[38;5;48m"
GREEN = "\033[38;5;82m"
VIOLET = "\033[38;5;177m"
MAGENTA = "\033[38;5;201m"
PURPLE = "\033[38;5;141m"
CRIMSON = "\033[38;5;197m"
AZURE = "\033[38;5;39m"
SLATE = "\033[38;5;244m"
DARK_SLATE = "\033[38;5;238m"
WHITE = "\033[38;5;255m"

# Sparkline Unicode Glyphs
BAR_GLYPHS = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
BRAILLE_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

CLUSTER_COLORS = [
    CYAN,
    EMERALD,
    GOLD,
    PURPLE,
    AZURE,
    MAGENTA,
    AMBER,
    WHITE,
]


def strip_ansi(text: str) -> str:
    """Removes ANSI escape codes to compute visible character length."""
    return re.sub(r"\033\[[0-9;]*m", "", text)


def pad_ansi(text: str, width: int, align: str = "left") -> str:
    """Pads ANSI formatted string to desired terminal column width."""
    visible_len = len(strip_ansi(text))
    pad = max(0, width - visible_len)
    if align == "right":
        return (" " * pad) + text
    if align == "center":
        left = pad // 2
        right = pad - left
        return (" " * left) + text + (" " * right)
    return text + (" " * pad)


def generate_sparkline(values: List[float], width: int = 24, color: str = CYAN) -> str:
    """Renders a dynamic ASCII/Unicode sparkline from a series of floats."""
    if not values:
        return f"{DARK_SLATE}{'─' * width}{RESET}"
    recent = values[-width:]
    if len(recent) < width:
        recent = [recent[0]] * (width - len(recent)) + recent

    min_v = min(recent)
    max_v = max(recent)
    span = max_v - min_v if (max_v - min_v) > 1e-6 else 1.0

    chars = []
    for v in recent:
        norm = max(0.0, min(1.0, (v - min_v) / span))
        idx = int(norm * (len(BAR_GLYPHS) - 1))
        chars.append(BAR_GLYPHS[idx])
    return f"{color}{''.join(chars)}{RESET}"


def build_projection_radar(
    vf: VectorForge,
    last_receipt: Optional[SearchReceipt] = None,
    grid_w: int = 54,
    grid_h: int = 15,
) -> List[str]:
    """Generates an ANSI 2D spatial projection radar grid."""
    # Find spatial bounds of projected 3D nodes
    nodes = vf.index.nodes
    if not nodes:
        return [f"  {SLATE}[No nodes in universe]{RESET}"]

    pts = []
    for nid, node in nodes.items():
        vx, vy, vz = node.visual_pos
        pts.append((vx, vz, nid, node.level, node.metadata.get("cluster", "")))

    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    span_x = max_x - min_x if (max_x - min_x) > 1e-4 else 1.0
    span_z = max_z - min_z if (max_z - min_z) > 1e-4 else 1.0

    # Initialize character grid
    # grid[y][x] = (char, color_code, priority)
    grid = [[("·", DARK_SLATE, 0) for _ in range(grid_w)] for _ in range(grid_h)]

    # Plot nodes
    cluster_color_map: Dict[str, str] = {}
    color_idx = 0

    for vx, vz, nid, lvl, cluster in pts:
        if cluster not in cluster_color_map:
            cluster_color_map[cluster] = CLUSTER_COLORS[color_idx % len(CLUSTER_COLORS)]
            color_idx += 1

        gx = int(max(0, min(grid_w - 1, ((vx - min_x) / span_x) * (grid_w - 1))))
        gy = int(max(0, min(grid_h - 1, ((vz - min_z) / span_z) * (grid_h - 1))))

        if lvl >= 2:
            char = "▲"
            clr = GOLD
            prio = 5
        elif lvl == 1:
            char = "○"
            clr = CYAN
            prio = 4
        else:
            char = "•"
            clr = cluster_color_map.get(cluster, EMERALD)
            prio = 2

        if prio > grid[gy][gx][2]:
            grid[gy][gx] = (char, clr, prio)

    # Plot trajectory hops if recent receipt exists
    top_k_ids = set()
    if last_receipt:
        for item in last_receipt.top_k:
            top_k_ids.add(item["id"])

        # Plot search hops
        for hop in last_receipt.trajectory:
            vpos = hop.get("visual_pos")
            if vpos:
                gx = int(max(0, min(grid_w - 1, ((vpos[0] - min_x) / span_x) * (grid_w - 1))))
                gy = int(max(0, min(grid_h - 1, ((vpos[2] - min_z) / span_z) * (grid_h - 1))))
                act = hop.get("action", "")
                if act == "entry_point":
                    grid[gy][gx] = ("★", GOLD, 8)
                elif act == "greedy_hop":
                    grid[gy][gx] = ("◆", AMBER, 6)

        # Highlight top-k nearest neighbors
        for rank, item in enumerate(last_receipt.top_k, 1):
            vpos = item.get("visual_pos")
            if vpos:
                gx = int(max(0, min(grid_w - 1, ((vpos[0] - min_x) / span_x) * (grid_w - 1))))
                gy = int(max(0, min(grid_h - 1, ((vpos[2] - min_z) / span_z) * (grid_h - 1))))
                symbol = str(rank) if rank <= 9 else "★"
                grid[gy][gx] = (symbol, f"{BOLD}{GOLD}", 10)

    # Convert grid to formatted strings
    output_lines = []
    top_border = f"{CYAN}┌" + ("─" * grid_w) + f"┐{RESET}"
    output_lines.append(top_border)

    for row in grid:
        line_chars = []
        current_clr = None
        for char, clr, _ in row:
            if clr != current_clr:
                line_chars.append(clr)
                current_clr = clr
            line_chars.append(char)
        line_chars.append(RESET)
        output_lines.append(f"{CYAN}│{RESET}" + "".join(line_chars) + f"{CYAN}│{RESET}")

    bot_border = f"{CYAN}└" + ("─" * grid_w) + f"┘{RESET}"
    output_lines.append(bot_border)
    return output_lines


def format_radar_dashboard(
    vf: VectorForge,
    last_receipt: Optional[SearchReceipt],
    rolling_latencies: List[float],
    rolling_evals: List[int],
    tick_count: int = 0,
) -> str:
    """Assembles the full multi-tier ANSI terminal HUD frame."""
    stats = vf.index.get_stats()
    spinner = BRAILLE_SPINNER[tick_count % len(BRAILLE_SPINNER)]

    lines: List[str] = []
    line_w = 98

    # Top Header
    title = f"{BOLD}{MAGENTA}⚡ VECTORFORGE // 15 FPS ANSI/BRAILLE HNSW TERMINAL RADAR ⚡{RESET}"
    status_tag = f"{EMERALD}● LIVE TELEMETRY{RESET} {spinner}"
    hdr = f" {title}  {pad_ansi(status_tag, line_w - len(strip_ansi(title)) - 4, 'right')}"
    lines.append(hdr)
    lines.append(f"{CYAN}{'═' * line_w}{RESET}")

    # Metrics Summary Bar
    ep = stats["entry_point"]
    ep_str = f"#{ep}" if ep is not None else "NONE"
    m_info = (
        f" {WHITE}DIM:{RESET} {CYAN}{stats['dimension']}D{RESET}  "
        f"│ {WHITE}METRIC:{RESET} {GOLD}{stats['metric'].upper()}{RESET}  "
        f"│ {WHITE}NODES:{RESET} {EMERALD}{stats['total_nodes']}{RESET}  "
        f"│ {WHITE}EDGES:{RESET} {AZURE}{stats['total_edges']}{RESET}  "
        f"│ {WHITE}MAX LAYER:{RESET} {PURPLE}L{stats['max_level']}{RESET}  "
        f"│ {WHITE}ENTRY POINT:{RESET} {GOLD}{ep_str}{RESET}  "
        f"│ {WHITE}M:{RESET} {stats['M']}  "
        f"│ {WHITE}efSearch:{RESET} {stats['ef_search']}"
    )
    lines.append(pad_ansi(m_info, line_w))
    lines.append(f"{DARK_SLATE}{'─' * line_w}{RESET}")

    # Split Screen: Left = Spatial Radar Grid, Right = Stratified Layers & Sparklines
    radar_lines = build_projection_radar(vf, last_receipt, grid_w=48, grid_h=13)

    # Right side content
    right_lines: List[str] = []
    right_lines.append(f"{BOLD}{PURPLE}STRATIFIED HIERARCHY TOPOLOGY:{RESET}")

    layer_dist = stats["layer_distribution"]
    total_n = max(1, stats["total_nodes"])

    for l_idx in range(len(layer_dist) - 1, -1, -1):
        count = layer_dist[l_idx]
        pct = (count / total_n) * 100.0
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        if l_idx >= 2:
            role = "Expressway"
            l_clr = GOLD
        elif l_idx == 1:
            role = "Highway"
            l_clr = CYAN
        else:
            role = "Dense Ground"
            l_clr = EMERALD

        right_lines.append(
            f"  {l_clr}[L{l_idx:02d}]{RESET} {role:<12} {count:>3} nodes ({pct:>5.1f}%) {l_clr}{bar}{RESET}"
        )

    right_lines.append("")
    right_lines.append(f"{BOLD}{CYAN}REAL-TIME METRIC SPARKLINES (LAST 30 PROBES):{RESET}")

    lat_spark = generate_sparkline(rolling_latencies, width=28, color=EMERALD)
    curr_lat = rolling_latencies[-1] if rolling_latencies else 0.0
    avg_lat = sum(rolling_latencies) / len(rolling_latencies) if rolling_latencies else 0.0
    right_lines.append(f"  Query Latency: {lat_spark}  {BOLD}{curr_lat:.3f}ms{RESET} (avg {avg_lat:.2f}ms)")

    eval_spark = generate_sparkline([float(x) for x in rolling_evals], width=28, color=AMBER)
    curr_eval = rolling_evals[-1] if rolling_evals else 0
    avg_eval = sum(rolling_evals) / len(rolling_evals) if rolling_evals else 0
    right_lines.append(f"  Dist Evals:    {eval_spark}  {BOLD}{curr_eval}{RESET} (avg {avg_eval:.0f})")

    right_lines.append("")
    right_lines.append(f"{BOLD}{GOLD}SPATIAL RADAR LEGEND:{RESET}")
    right_lines.append(
        f"  {GOLD}▲{RESET} Layer 2+ Express  {CYAN}○{RESET} Layer 1 Highway  {EMERALD}•{RESET} Layer 0 Ground  {GOLD}★{RESET} Entry/Top-1"
    )

    # Stitch Left and Right side by side
    max_h = max(len(radar_lines), len(right_lines))
    for i in range(max_h):
        r_part = radar_lines[i] if i < len(radar_lines) else " " * 50
        r_info = right_lines[i] if i < len(right_lines) else ""
        lines.append(f" {r_part}  {r_info}")

    lines.append(f"{DARK_SLATE}{'─' * line_w}{RESET}")

    # Bottom Trajectory & Nearest Neighbors Inspection
    if last_receipt:
        rec = last_receipt
        rec_hdr = (
            f" {BOLD}{GOLD}LATEST k-NN SEARCH RECEIPT{RESET} │ "
            f"Latency: {EMERALD}{rec.latency_ms:.3f} ms{RESET} │ "
            f"Evaluations: {AMBER}{rec.distance_evaluations}{RESET} │ "
            f"Routing Hops: {CYAN}{rec.hop_count}{RESET} │ "
            f"Entry: #{rec.entry_point} (L{rec.entry_level})"
        )
        lines.append(rec_hdr)

        # Trajectory path summary
        hops_str = []
        for step in rec.trajectory[:8]:
            act = step["action"]
            lyr = step["layer"]
            tn = step["to_node"]
            if act == "entry_point":
                hops_str.append(f"{GOLD}[L{lyr}:#{tn}]{RESET}")
            elif act == "greedy_hop":
                hops_str.append(f"{AMBER}➔#{tn}{RESET}")
            elif act == "layer_descent":
                hops_str.append(f"{PURPLE}↓L{lyr}{RESET}")
            elif act == "beam_explore":
                hops_str.append(f"{CYAN}⠶#{tn}{RESET}")
        if len(rec.trajectory) > 8:
            hops_str.append(f"{SLATE}... (+{len(rec.trajectory) - 8} hops){RESET}")
        lines.append(f"  {WHITE}Trajectory Vector:{RESET} " + " ".join(hops_str))

        # Top Nearest Neighbors table
        lines.append(
            f"  {BOLD}{'RANK':<5} {'NODE':<7} {'DISTANCE':<10} {'SIMILARITY':<12} {'CLUSTER':<24} {'TOKEN':<12}{RESET}"
        )
        for rank, item in enumerate(rec.top_k[:5], 1):
            meta = item.get("metadata", {})
            cluster = meta.get("cluster", "Unknown")
            token = meta.get("token", f"VEC-{item['id']:04d}")
            sim_bar = "█" * int(item["similarity"] * 10)
            row = (
                f"  {GOLD}#{rank:<4}{RESET} "
                f"#{item['id']:<6} "
                f"{item['distance']:<10.4f} "
                f"{EMERALD}{item['similarity'] * 100.0:>5.1f}% {sim_bar:<10}{RESET} "
                f"{CYAN}{cluster:<24}{RESET} "
                f"{token:<12}"
            )
            lines.append(row)
    else:
        lines.append(f"  {SLATE}[No query executed yet. Press SPACE to inject a routing probe]{RESET}")

    lines.append(f"{CYAN}{'═' * line_w}{RESET}")
    lines.append(
        f" {DARK_SLATE}[SPACE] Fire Query Probe  [B] Run Benchmark Sweep  [+] Increase efSearch  [-] Decrease  [Q] Exit{RESET}"
    )

    return "\n".join(lines)


def run_terminal_radar(
    vf: VectorForge,
    target_fps: float = 15.0,
    auto_probe_interval: float = 1.0,
) -> None:
    """Launches interactive 15 FPS ANSI/Braille live terminal HUD."""
    refresh_rate = 1.0 / target_fps
    tick = 0
    last_probe_time = 0.0

    rolling_latencies: List[float] = [0.24, 0.28, 0.31, 0.22, 0.25]
    rolling_evals: List[int] = [32, 41, 38, 29, 35]
    last_receipt: Optional[SearchReceipt] = None

    # Trigger initial query
    if vf.index.nodes:
        sample_node = random.choice(list(vf.index.nodes.values()))
        probe = [x + random.gauss(0.0, 0.15) for x in sample_node.vector]
        last_receipt = vf.search(query=probe, k=5)
        rolling_latencies.append(last_receipt.latency_ms)
        rolling_evals.append(last_receipt.distance_evaluations)

    # Terminal cursor hide and alternate screen
    sys.stdout.write("\033[?25l")  # Hide cursor
    sys.stdout.flush()

    try:
        while True:
            now = time.time()

            # Automatic probe trigger
            if now - last_probe_time >= auto_probe_interval and vf.index.nodes:
                sample_node = random.choice(list(vf.index.nodes.values()))
                probe = [x + random.gauss(0.0, 0.15) for x in sample_node.vector]
                last_receipt = vf.search(query=probe, k=5)
                rolling_latencies.append(last_receipt.latency_ms)
                if len(rolling_latencies) > 30:
                    rolling_latencies.pop(0)
                rolling_evals.append(last_receipt.distance_evaluations)
                if len(rolling_evals) > 30:
                    rolling_evals.pop(0)
                last_probe_time = now

            # Non-blocking key check on stdin if tty
            if sys.stdin.isatty():
                r, _, _ = select.select([sys.stdin], [], [], 0.0)
                if r:
                    ch = sys.stdin.read(1).lower()
                    if ch == "q":
                        break
                    elif ch == " ":
                        sample_node = random.choice(list(vf.index.nodes.values()))
                        probe = [x + random.gauss(0.0, 0.2) for x in sample_node.vector]
                        last_receipt = vf.search(query=probe, k=5)
                        rolling_latencies.append(last_receipt.latency_ms)
                        rolling_evals.append(last_receipt.distance_evaluations)
                    elif ch == "+":
                        vf.index.ef_search = min(256, vf.index.ef_search + 16)
                    elif ch == "-":
                        vf.index.ef_search = max(8, vf.index.ef_search - 16)
                    elif ch == "b":
                        vf.benchmark_recall(num_queries=15, k=5)

            # Draw frame
            frame = format_radar_dashboard(
                vf=vf,
                last_receipt=last_receipt,
                rolling_latencies=rolling_latencies,
                rolling_evals=rolling_evals,
                tick_count=tick,
            )

            # Move cursor to home and write
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(frame + "\n")
            sys.stdout.flush()

            tick += 1
            time.sleep(refresh_rate)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")  # Show cursor
        sys.stdout.flush()
        print(f"{EMERALD}⚡ [VectorForge] Terminal Radar stopped.{RESET}")
