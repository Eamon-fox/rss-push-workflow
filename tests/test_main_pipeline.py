import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import main
from src.models import NewsItem


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


class TestMainPipeline(unittest.TestCase):
    def test_run_smoke_without_network_or_llm(self):
        items = [
            NewsItem(title="Correction: something", content="cell", source_name="Some Journal"),
            NewsItem(title="Quantum gravity paper", content="black holes", source_name="Physics Today"),
            NewsItem(title="A note on RTCB", content="short", source_name="Some Journal"),
            NewsItem(title="ER stress response", content="in neurons", source_name="Some Journal"),
        ]

        real_batch_clean = main.clean.batch_clean
        real_filter_unseen = main.dedup.filter_unseen

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            seen_path = td_path / "seen.json"

            def _load_seen():
                if not seen_path.exists():
                    return {}
                return json.loads(seen_path.read_text(encoding="utf-8"))

            def _save_seen(seen: dict, _filepath: str = "data/seen.json"):
                seen_path.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")

            real_filter_hybrid = importlib.import_module("src.4_filter.hybrid").filter_hybrid
            hybrid_mod = importlib.import_module("src.4_filter.hybrid")

            def _filter_hybrid_no_save(*args, **kwargs):
                kwargs["save"] = False
                return real_filter_hybrid(*args, **kwargs)

            def _fake_llm_process_batch(llm_items, score_threshold=6.0):
                out: list[NewsItem] = []
                for it in llm_items:
                    out.append(it.model_copy(update={"score": score_threshold + 1.0, "summary": "ok"}))
                return out

            with (
                patch.object(main.aggregate, "fetch_all", return_value=items),
                patch.object(main.clean, "batch_clean", side_effect=lambda xs: real_batch_clean(xs, save=False)),
                patch.object(main.dedup, "load", side_effect=_load_seen),
                patch.object(main.dedup, "save", side_effect=_save_seen),
                patch.object(main.dedup, "filter_unseen", side_effect=lambda xs, seen: real_filter_unseen(xs, seen, save=False)),
                patch.object(hybrid_mod, "_default_embedder", return_value=DummyEmbedder()),
                patch.object(main.filt, "filter_hybrid", side_effect=_filter_hybrid_no_save),
                patch.object(main.llm, "process_batch", side_effect=_fake_llm_process_batch),
                patch.object(main.deliver, "to_console", return_value=None),
                patch.object(main.deliver, "to_json", return_value=None),
            ):
                main.run()

            seen = _load_seen()
            # All 4 items have a fingerprint (title hash) -> should be marked.
            self.assertEqual(len(seen), 4)


if __name__ == "__main__":
    unittest.main()
