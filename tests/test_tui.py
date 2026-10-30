"""Unit tests for VectorForge ANSI Terminal Radar HUD."""

import unittest
from vectorforge.index import VectorForge
from vectorforge.tui import (
    strip_ansi,
    pad_ansi,
    generate_sparkline,
    build_projection_radar,
    format_radar_dashboard,
)


class TestTerminalRadar(unittest.TestCase):
    """Test suite for TUI HUD formatting and spatial radar rendering."""

    def setUp(self):
        self.vf = VectorForge(dim=8, M=8, ef_construction=32, ef_search=16)
        self.vf.seed_synthetic_universe(num_vectors=30, num_clusters=3, seed=42)

    def test_strip_and_pad_ansi(self):
        colored_text = "\033[1;36mVECTORFORGE\033[0m"
        stripped = strip_ansi(colored_text)
        self.assertEqual(stripped, "VECTORFORGE")

        padded_left = pad_ansi(colored_text, 15, align="left")
        self.assertEqual(len(strip_ansi(padded_left)), 15)

        padded_right = pad_ansi(colored_text, 15, align="right")
        self.assertEqual(len(strip_ansi(padded_right)), 15)

    def test_generate_sparkline(self):
        values = [0.1, 0.2, 0.5, 0.8, 1.2]
        spark = generate_sparkline(values, width=10)
        stripped = strip_ansi(spark)
        self.assertEqual(len(stripped), 10)

    def test_build_projection_radar(self):
        radar_lines = build_projection_radar(self.vf, grid_w=30, grid_h=10)
        self.assertTrue(len(radar_lines) >= 10)
        # Check border formatting
        self.assertIn("┌", radar_lines[0])
        self.assertIn("└", radar_lines[-1])

    def test_format_radar_dashboard(self):
        # Execute query
        receipt = self.vf.search(query=self.vf.index.nodes[0].vector, k=3)
        dashboard = format_radar_dashboard(
            vf=self.vf,
            last_receipt=receipt,
            rolling_latencies=[0.2, 0.3, 0.25],
            rolling_evals=[20, 25, 22],
            tick_count=1,
        )
        self.assertIn("VECTORFORGE", dashboard)
        self.assertIn("DIM:", dashboard)
        self.assertIn("STRATIFIED HIERARCHY TOPOLOGY", dashboard)
        self.assertIn("LATEST k-NN SEARCH RECEIPT", dashboard)


if __name__ == "__main__":
    unittest.main()
