import unittest
import importlib

import numpy as np

from src.models import NewsItem

filter_hybrid = importlib.import_module("src.4_filter.hybrid").filter_hybrid


class DummyEmbedder:
    def encode(self, texts: list[str]) -> object:
        vecs: list[np.ndarray] = []
        for text in texts:
            t = (text or "").lower()
            v = np.zeros(3, dtype=np.float32)
            if "rtcb" in t or "rna ligation" in t:
                v[0] = 1.0
            if "ire1" in t or "xbp1" in t or "er stress" in t or "upr" in t:
                v[1] = 1.0
            if "neurodegeneration" in t or "neuron" in t:
                v[2] = 1.0
            if float(v.sum()) == 0.0:
                v[0] = 0.01
            v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.stack(vecs, axis=0)


class TestHybridFilter(unittest.TestCase):
    def test_layer1_blacklist_and_coarse_bio(self):
        items = [
            NewsItem(
                title="Retraction: A quantum paper",
                content="irrelevant",
                source_name="Some Journal",
            ),
            NewsItem(
                title="Quantum gravity in condensed matter",
                content="black holes and galaxies",
                source_name="Physics Today",
            ),
            NewsItem(
                title="ER stress response in neurons",
                content="unfolded protein response and signaling",
                source_name="Some Journal",
            ),
        ]

        kept, dropped, stats = filter_hybrid(items, embedder=DummyEmbedder(), save=False)
        self.assertEqual(stats.total, 3)
        self.assertEqual(stats.layer1_dropped, 2)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 2)

    def test_layer2_vip_bypass(self):
        items = [
            NewsItem(
                title="A short note on RTCB",
                content="no other context",
                source_name="Some Journal",
            ),
        ]
        kept, dropped, stats = filter_hybrid(items, embedder=DummyEmbedder(), save=False)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 0)
        self.assertEqual(stats.layer2_vip_kept, 1)

    def test_top_journal_bypass_layer1(self):
        items = [
            NewsItem(
                title="A news brief",
                content="Neurodegeneration and RNA ligation in neurons",
                source_name="Nature",
            ),
        ]
        kept, dropped, stats = filter_hybrid(items, embedder=DummyEmbedder(), save=False)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats.layer1_dropped, 0)


if __name__ == "__main__":
    unittest.main()
