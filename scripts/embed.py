"""Embed every ticket once.

Full run (team model, no --sample) also writes the team's SHARED index (README "Shared embeddings file"):
  data/index/emb.npy  (22869, 768) float32, normalized, rows in load_split()[0] order
  data/index/kb.pkl   same row order: ticket_id, text, subject, answer, language, type, queue, priority,
                      tag_1..tag_8, searchable
  data/index/meta.json  model + prefix, so retrieval can check it embeds queries the same way
Triage saves the same matrix as artifacts/embeddings/<model>/kb.npy (train_triage.py cuts the
family-based train / half_a / half_b from it); the holdout (test_inputs) is embedded separately and
never enters the index.
"""
import json
import time

import numpy as np
import pandas as pd
from _common import REPO_ROOT, parser, setup_logging

from sentineldesk.triage.config import artifacts_dir, embedding_prefix, load_config, model_slug
from sentineldesk.triage.data import assert_no_test_ids, ensure_csv, kb_index_frame, read_part
from sentineldesk.triage.embed import Embedder
from sentineldesk.triage.text import build_texts
from split import load_split



def save_part(out, part, X, ids):
    np.save(out / f"{part}.npy", X.astype(np.float32))
    (out / f"{part}_ids.json").write_text(json.dumps(list(ids)))
    print(f"{part}: {X.shape}")


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--model", default=None, help="sentence-transformers model (default from config)")
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    root = artifacts_dir(cfg, args.artifacts, args.sample)
    team_model = cfg["embedding"]["model"]
    model = args.model or team_model
    prefix = embedding_prefix(cfg, model)
    emb = Embedder(model, prefix, args.batch_size or cfg["embedding"]["batch_size"], cfg["embedding"]["max_seq_length"])
    out = root / "embeddings" / model_slug(model)
    out.mkdir(parents=True, exist_ok=True)
    kb_part = read_part(root / "splits", "kb")
    test_in = read_part(root / "splits", "test_inputs")
    test_ids = test_in["ticket_id"]
    assert_no_test_ids(kb_part["ticket_id"], test_ids)

    if args.sample:  # quick run: embed the sampled kb directly, never touch the shared index
        X = emb.encode(build_texts(kb_part["subject"], kb_part["body"]), show_progress=True)
        save_part(out, "kb", X, kb_part["ticket_id"])
    else:
        kb, _ = load_split(str(ensure_csv(cfg["data"].get("csv_path"))))
        kb = kb.reset_index(drop=True)
        if kb["ticket_id"].tolist() != kb_part["ticket_id"].tolist():
            raise AssertionError("artifacts/splits/kb.parquet does not match load_split(); re-run scripts/build_split.py")
        texts = build_texts(kb["subject"], kb["body"])
        if texts != kb["text"].tolist():
            raise AssertionError("build_text no longer matches split.py's `text` column; the shared index would drift")
        t0 = time.perf_counter()
        E = emb.encode(texts, show_progress=True)
        print(f"kb: {E.shape} in {time.perf_counter() - t0:.1f}s")
        save_part(out, "kb", E, kb["ticket_id"])
        if model == team_model:
            idx_dir = REPO_ROOT / cfg["paths"].get("index_dir", "data/index")
            idx_dir.mkdir(parents=True, exist_ok=True)
            np.save(idx_dir / "emb.npy", E.astype(np.float32))
            frame = kb_index_frame(kb)
            pd.to_pickle(frame, idx_dir / "kb.pkl")
            (idx_dir / "meta.json").write_text(json.dumps(
                {"embedding_model": model, "prefix": prefix, "normalize_embeddings": True, "dim": int(E.shape[1]),
                 "rows": int(E.shape[0]), "searchable": int(frame["searchable"].sum())}, indent=2))
            print(f"shared index: {idx_dir}/emb.npy {E.shape}, kb.pkl ({int(frame['searchable'].sum())} searchable)")
        else:
            print(f"NOT writing data/index: {model} is not the team model ({team_model})")

    df = test_in
    t0 = time.perf_counter()
    save_part(out, "test_inputs", emb.encode(build_texts(df["subject"], df["body"]), show_progress=True), test_ids)
    print(f"test_inputs embedded in {time.perf_counter() - t0:.1f}s (holdout: never part of the index)")
    (out / "meta.json").write_text(json.dumps({"embedding_model": model, "prefix": prefix, "dim": emb.dim,
                                               "batch_size": emb.batch_size,
                                               "max_seq_length": cfg["embedding"]["max_seq_length"]}, indent=2))


if __name__ == "__main__":
    main()
