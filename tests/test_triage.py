"""Triage tests. Synthetic data + a hashing "embedder" keep them offline and fast (< 1 min)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sentineldesk.triage import TASKS  # noqa: E402
from sentineldesk.triage.config import load_config  # noqa: E402
from sentineldesk.triage.data import (TEST_INPUT_COLS, assert_disjoint, build_parts,  # noqa: E402
                                      read_part, read_test_labels, write_parts)
from sentineldesk.triage.keywords import KeywordMatcher  # noqa: E402
from sentineldesk.triage.labels import load_labels  # noqa: E402
from sentineldesk.triage.neighbours import NeighbourIndex  # noqa: E402
from sentineldesk.triage.text import build_text, build_texts  # noqa: E402
from sentineldesk.triage.training import train_all  # noqa: E402


# ---------- fixtures ---------------------------------------------------------------------------
class FakeEmbedder:
    """Deterministic bag-of-hashed-words embedding (normalized), so tests need no model download."""

    def __init__(self, dim: int = 128):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.hv = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")

    def encode(self, texts, show_progress=False):
        X = self.hv.transform(texts).toarray().astype(np.float32)
        X[np.linalg.norm(X, axis=1) == 0, 0] = 1.0
        return X


TOPICS = {
    ("Incident", "Technical Support", "high"): ("Server outage", "Our EMR system is down, kein Zugriff for all users since morning."),
    ("Incident", "IT Support", "medium"): ("Laptop crashed", "My laptop crashed after the update and the screen stays black."),
    ("Request", "Billing and Payments", "low"): ("Invoice question", "Please send a copy of the last invoice for our accounting team."),
    ("Request", "Sales and Pre-Sales", "medium"): ("Pricing", "What does the enterprise plan cost for fifty seats?"),
    ("Problem", "Product Support", "medium"): ("Sync problem", "Die Synchronisierung funktioniert nicht zuverlässig mit dem Kalender."),
    ("Change", "IT Support", "low"): ("Upgrade request", "Please upgrade our firewall firmware to the latest version next week."),
}


def synthetic_df(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = list(TOPICS)
    rows = []
    for i in range(n):
        k = keys[rng.integers(len(keys))]
        subj, body = TOPICS[k]
        subject = [subj, None, float("nan"), ""][rng.integers(4)]
        rows.append({"subject": subject, "body": f"{body} ref {rng.integers(1000)} filler{rng.integers(30)}",
                     "answer": "Thanks <name>, we are on it.", "type": k[0], "queue": k[1], "priority": k[2],
                     "language": "en", "tag_1": "Outage" if k[2] == "high" else "Bug"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "tickets.csv"
    synthetic_df().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def parts(csv_path):
    """Team split (kb / test), then the family-based train / half_a / half_b, as train_triage.py does."""
    from sentineldesk.triage.data import family_calib_split

    base = build_parts(csv_path)
    kb = base["kb"].reset_index(drop=True)
    emb = FakeEmbedder()
    E = emb.encode(build_texts(kb["subject"], kb["body"]))
    pos, fam = family_calib_split(kb, E, 0.98)
    out = {k: kb.iloc[v].reset_index(drop=True) for k, v in pos.items()}
    out.update(kb=kb, test=base["test"], _embs={k: E[v] for k, v in pos.items()}, _fam=fam, _pos=pos)
    return out


@pytest.fixture(scope="module")
def trained(tmp_path_factory, parts):
    root = tmp_path_factory.mktemp("artifacts")
    cfg = load_config()
    cfg["model_tfidf"]["use_char"] = False
    emb = FakeEmbedder()
    meta = {"embedding_model": "fake", "prefix": "", "dim": 128}
    train_all(cfg, {k: parts[k] for k in ("train", "half_a", "half_b")}, parts["_embs"], root / "triage", meta,
              test_ids=parts["test"]["ticket_id"].tolist())
    write_parts({"kb": parts["kb"], "test": parts["test"]}, root / "splits")
    return root, emb


def load_triage(root, emb):
    from sentineldesk.triage.predict import Triage

    return Triage.load(root, embedder=emb)


# ---------- text ---------------------------------------------------------------------------------
@pytest.mark.parametrize("subject", [None, float("nan"), ""])
def test_build_text_missing_subject(subject):
    out = build_text(subject, "VPN is down")
    assert out == "\nVPN is down" and "nan" not in out


def test_build_text_raises_on_empty():
    with pytest.raises(ValueError):
        build_text(None, float("nan"))
    with pytest.raises(ValueError):
        build_text("", "  ")


def test_build_text_matches_team_split_text(csv_path):
    """The shared index is built from split.py's `text`; triage must embed the identical string."""
    from split import load_split

    kb, _ = load_split(str(csv_path))
    assert build_texts(kb["subject"], kb["body"]) == kb["text"].tolist()


def test_searchable_flag():
    from sentineldesk.triage.data import is_searchable

    assert not is_searchable(None) and not is_searchable(float("nan")) and not is_searchable("")
    assert not is_searchable("Thanks, we fixed it.")                       # under 40 chars
    assert not is_searchable("<name> <acc_num> <tel_num> <email> <name> <name>")  # placeholders only
    assert is_searchable("Dear <name>, please restart the VPN client and clear the cache, then retry.")


# ---------- split / leakage ---------------------------------------------------------------------
def test_split_reproducible_disjoint_and_complete(parts, csv_path):
    again = build_parts(csv_path)
    for k in ("kb", "test"):
        assert parts[k]["ticket_id"].tolist() == again[k]["ticket_id"].tolist()
    all_ids = [i for k in ("train", "half_a", "half_b", "test") for i in parts[k]["ticket_id"]]
    assert len(all_ids) == len(set(all_ids)) == 900
    assert "answer" not in parts["test"].columns


def test_parts_follow_team_split(parts, csv_path):
    from split import load_split

    kb, test = load_split(str(csv_path))
    assert parts["test"]["ticket_id"].tolist() == test["ticket_id"].tolist()
    assert parts["kb"]["ticket_id"].tolist() == kb["ticket_id"].tolist()
    triage_kb = set().union(*(parts[k]["ticket_id"] for k in ("train", "half_a", "half_b")))
    assert triage_kb == set(kb["ticket_id"])


def test_family_split_keeps_reworded_copies_together():
    """Audit items 3 + 4: no paraphrase family may span train and the calibration halves."""
    from sentineldesk.triage.data import family_calib_split, paraphrase_families

    rng = np.random.default_rng(0)
    base = rng.normal(size=(150, 16))
    E = np.repeat(base, 4, axis=0) + rng.normal(scale=1e-3, size=(600, 16))  # 150 families x 4 copies
    E = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
    fam = paraphrase_families(E, 0.98)
    assert len(set(fam)) == 150
    pos, fam2 = family_calib_split(pd.DataFrame({"ticket_id": range(600)}), E, 0.98)
    assert sorted(np.concatenate(list(pos.values()))) == list(range(600))
    for a, b in (("train", "half_a"), ("train", "half_b"), ("half_a", "half_b")):
        assert not set(fam2[pos[a]]) & set(fam2[pos[b]])
    assert len(pos["half_a"]) > 0 and len(pos["half_b"]) > 0


def test_assert_disjoint_catches_leak():
    with pytest.raises(AssertionError):
        assert_disjoint({"train": ["TKT-00001"], "test": ["TKT-00001"]})


def test_test_files_columns(trained):
    root, _ = trained
    assert list(pd.read_parquet(root / "splits" / "test_inputs.parquet").columns) == TEST_INPUT_COLS
    assert list(pd.read_parquet(root / "splits" / "test_labels.parquet").columns) == ["ticket_id", "type", "queue", "priority"]
    with pytest.raises(PermissionError):
        read_part(root / "splits", "test_labels")
    with pytest.raises(PermissionError):
        read_test_labels(root / "splits", root / "does_not_exist.parquet")


def test_no_holdout_id_in_index(trained, parts):
    t = load_triage(*trained)
    assert not set(t.nn.ids) & set(parts["test"]["ticket_id"])


def test_real_split_files_if_present():
    p = REPO_ROOT / "artifacts" / "splits" / "test_inputs.parquet"
    if not p.exists():
        pytest.skip("full split not built yet")
    assert list(pd.read_parquet(p).columns) == TEST_INPUT_COLS


# ---------- neighbours ----------------------------------------------------------------------------
def test_neighbour_self_exclusion():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(50, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = [f"TKT-{i:05d}" for i in range(50)]
    labels = {t: ["a", "b"] for t in TASKS}
    codes = {t: rng.integers(2, size=50) for t in TASKS}
    nn = NeighbourIndex(emb, ids, codes, labels, k=5)
    S, I = nn.search(emb[:10], ids[:10])
    for r in range(10):
        assert ids[r] not in [ids[i] for i in I[r]]
        assert len(I[r]) == 5
    probs = nn.vote(S, I)
    assert np.allclose(probs["type"].sum(1), 1)


# ---------- keywords ------------------------------------------------------------------------------
def test_keyword_matching():
    kw = KeywordMatcher(["down", "kein Zugriff", "Störung", "can't access"])
    assert kw.matches("please download the file")[0] == []
    assert kw.matches("The server is DOWN!")[0] == ["down"]
    assert kw.matches("Wir haben kein   Zugriff mehr")[0] == ["kein zugriff"]
    assert kw.matches("kein Zugriffsrecht")[0] == []
    assert kw.matches("Große STÖRUNG im Netz")[0] == ["störung"]
    assert kw.matches("I can’t access email")[0] == ["can't access"]


def test_feature_cleaning_literal_newlines_and_br():
    from sentineldesk.triage.text import clean_for_features

    assert clean_for_features("Hi\\nserver<br>down<BR/>now") == "Hi\nserver\ndown\nnow"
    assert KeywordMatcher(["down"]).matches("server\\ndown")[0] == ["down"]   # literal \n no longer glues words


def test_keyword_yaml_loads():
    kw = KeywordMatcher.from_yaml(REPO_ROOT / "keywords" / "outage_keywords.yaml")
    assert "ausfall" in kw.keywords and "outage" in kw.keywords


# ---------- predict -------------------------------------------------------------------------------
REQUIRED = {"mode", "type", "queue", "priority", "critical", "neighbours", "suggested_conf_thresholds", "latency_ms"}


def _check_output(out, labels):
    assert REQUIRED <= set(out)
    for t in TASKS:
        assert list(out[t]["probs"]) == labels[t]
        assert math.isclose(sum(out[t]["probs"].values()), 1.0, abs_tol=1e-6)
        assert out[t]["label"] == max(out[t]["probs"], key=out[t]["probs"].get)
    assert {"score", "threshold", "fires", "keyword_hits"} <= set(out["critical"])
    assert len(out["neighbours"]["top"]) <= 5 and set(out["neighbours"]["explanation"]) == set(TASKS)
    assert "total" in out["latency_ms"]


def test_predict_full(trained):
    root, emb = trained
    t = load_triage(root, emb)
    labels = load_labels(root / "triage" / "labels.json")
    assert t.mode == "full"
    calib = json.loads((root / "triage" / "calibration.json").read_text())
    for task in TASKS:
        w = calib["full"][task]["pool_weights_m0_m1_m2"]
        assert len(w) == 3 and math.isclose(sum(w), 1.0, abs_tol=1e-6)
    en = t.predict("Server outage", "Our EMR system is down for all users")
    _check_output(en, labels)
    assert "emr" in en["critical"]["keyword_hits"]
    _check_output(t.predict("Störung", "Die Synchronisierung funktioniert nicht"), labels)
    _check_output(t.predict(None, "Please send the last invoice"), labels)
    _check_output(t.predict("", "Please send the last invoice"), labels)
    with pytest.raises(ValueError):
        t.predict("", "")
    assert len(t.predict_batch(["a", None], ["invoice please", "server down"])) == 2


def test_fallback_mode(trained, tmp_path):
    import shutil

    root, emb = trained
    shutil.copytree(root / "triage", tmp_path / "triage")
    (tmp_path / "triage" / "ensemble.joblib").unlink()
    t = load_triage(tmp_path, emb)
    assert t.mode == "fallback"
    _check_output(t.predict("Invoice", "Please send a copy of the invoice"), load_labels(tmp_path / "triage" / "labels.json"))


def test_temperature_scaling_does_not_hurt_nll(trained):
    root, _ = trained
    calib = json.loads((root / "triage" / "calibration.json").read_text())
    for mode in ("fallback", "full"):
        for t in TASKS:
            assert calib[mode][t]["nll_after"] <= calib[mode][t]["nll_before"] + 1e-9


def test_neighbours_never_import_faiss():
    """FAISS + PyTorch on macOS: segfault/hang when FAISS loads after the embedder. Keep it out."""
    import subprocess

    code = ("import sys, numpy as np; sys.path.insert(0, %r)\n"
            "from sentineldesk.triage.neighbours import NeighbourIndex\n"
            "e = np.eye(4, dtype=np.float32); nn = NeighbourIndex(e, list('abcd'), {t: np.zeros(4, int) for t in "
            "('type','queue','priority')}, {t: ['x'] for t in ('type','queue','priority')}, k=2)\n"
            "nn.search(e, list('abcd')); assert 'faiss' not in sys.modules, 'faiss was imported'") % str(REPO_ROOT)
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


@pytest.mark.parametrize("method", ["pool", "stack"])
def test_ensemble_methods(method):
    from sentineldesk.triage.ensemble import Ensemble

    rng = np.random.default_rng(0)
    labels = {t: ["a", "b", "c"] for t in TASKS}
    y = {t: rng.integers(3, size=200) for t in TASKS}

    def noisy(acc):  # a model whose top class is the true label with probability acc, else a wrong one
        out = {}
        for t in TASKS:
            right = rng.random(200) < acc
            top = np.where(right, y[t], (y[t] + rng.integers(1, 3, size=200)) % 3)
            p = np.full((200, 3), 0.15)
            p[np.arange(200), top] = 0.7
            out[t] = p
        return out

    ps = [noisy(0.4), noisy(0.8), noisy(0.5)]
    ens = Ensemble(method=method).fit(ps, y, labels)
    pr = ens.predict_proba(ps, labels)
    for t in TASKS:
        assert pr[t].shape == (200, 3) and np.allclose(pr[t].sum(1), 1)
        if method == "pool":
            assert max(range(3), key=lambda i: ens.weights[t][i]) == 1   # trusts the best model most
