"""Embed every ticket once.

Full run (team model, no --sample) also writes the team's SHARED index (README "Shared embeddings file"):
  data/index/emb.npy  (22869, 768) float32, normalized, rows in load_split()[0] order
  data/index/kb.pkl   same row order: ticket_id, text, subject, answer, language, type, queue, priority,
                      tag_1..tag_8, searchable
  data/index/meta.json  model + prefix, so retrieval can check it embeds queries the same way
Triage's per-part arrays (train / half_a / half_b) are sliced from that same matrix; the holdout
(test_inputs) is embedded separately and never enters the index.
"""
import json
import time

import numpy as np
import pandas as pd
from _common import REPO_ROOT, parser, setup_logging

from sentineldesk.triage.config import artifacts_dir, embedding_prefix, load_config, model_slug
from sentineldesk.triage import TASKS
from sentineldesk.triage.data import (TRAIN_COLS, assert_no_test_ids, ensure_csv, family_calib_split,
                                      kb_index_frame, paraphrase_families, read_part)
from sentineldesk.triage.embed import Embedder
from sentineldesk.triage.text import build_texts
from split import load_split

PARTS = ("train", "half_a", "half_b")


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
    parts = {k: read_part(root / "splits", k) for k in (*PARTS, "test_inputs")}
    test_ids = parts["test_inputs"]["ticket_id"]

    if args.sample:  # quick run: embed the sampled parts directly, never touch the shared index
        for part in PARTS:
            df = parts[part]
            save_part(out, part, emb.encode(build_texts(df["subject"], df["body"]), show_progress=True), df["ticket_id"])
    else:
        kb, _ = load_split(str(ensure_csv(cfg["data"].get("csv_path"))))
        kb = kb.reset_index(drop=True)
        texts = build_texts(kb["subject"], kb["body"])
        if texts != kb["text"].tolist():
            raise AssertionError("build_text no longer matches split.py's `text` column; the shared index would drift")
        part_ids = {k: parts[k]["ticket_id"].tolist() for k in PARTS}
        if set(kb["ticket_id"]) != set().union(*part_ids.values()):
            raise AssertionError("artifacts/splits does not match load_split(); re-run scripts/build_split.py")
        assert_no_test_ids(kb["ticket_id"], test_ids)
        t0 = time.perf_counter()
        E = emb.encode(texts, show_progress=True)
        print(f"kb: {E.shape} in {time.perf_counter() - t0:.1f}s")

        # Data audit #3/#4: redraw halves A/B by whole paraphrase family (needs these embeddings, so it
        # happens here, not in build_split.py) and overwrite the random calibration parts. Only for the
        # team model: embedding another model for comparison must not change the team's split.
        # ponytail: --sample runs keep the random split (smoke test only); run build_split.py BEFORE this.
        if model == team_model:
            thr = cfg["calibration"].get("family_threshold", 0.98)
            family, top1, nn1 = paraphrase_families(E, thr)
            sizes = np.bincount(family)
            print(f"paraphrase families @ cosine>={thr}: {len(sizes)} families, largest {sizes.max()}, "
                  f"{(sizes[family] > 1).mean():.1%} of KB tickets have a reworded copy")
            band_queue = None
            for lo, hi in ((0.99, 1.01), (thr, 0.99), (0.95, thr), (-1, 0.95)):  # labels vs nearest KB ticket
                m = (top1 >= lo) & (top1 < hi)
                if m.any():
                    agree = {t: (kb[t].to_numpy()[m] == kb[t].to_numpy()[nn1[m]]).mean() for t in TASKS}
                    band_queue = agree["queue"] if lo == thr else band_queue
                    print(f"  nearest-KB cosine [{lo:.2f}, {min(hi, 1):.2f}): {m.sum()} tickets, same labels "
                          + ", ".join(f"{t} {v:.1%}" for t, v in agree.items()))
            # A real reworded copy shares its labels ~97-100% of the time (e5-base measured 96.9% queue in
            # [0.98, 0.99)); related-but-different tickets only ~63%. Below 95%, families are too broad.
            if sizes.max() > 0.02 * len(kb) or (band_queue is not None and band_queue < 0.95):
                raise SystemExit(f"family_threshold {thr} is too low for {model} (largest family {sizes.max()}, "
                                 f"queue agreement just above it {band_queue}); raise it in configs/triage.yaml")
            fam_parts = dict(zip(PARTS, family_calib_split(kb, family)))
            for part, df in fam_parts.items():
                df[[c for c in TRAIN_COLS if c in df.columns]].to_parquet(root / "splits" / f"{part}.parquet",
                                                                           index=False)
            part_ids = {k: v["ticket_id"].tolist() for k, v in fam_parts.items()}
            (root / "splits" / "sizes.json").write_text(json.dumps(
                {**{k: len(v) for k, v in part_ids.items()}, "test": len(test_ids), "family_threshold": thr}, indent=2))
            print("family-aware calibration parts: " + ", ".join(f"{k}={len(v)}" for k, v in part_ids.items()))

        row = {tid: i for i, tid in enumerate(kb["ticket_id"])}
        for part in PARTS:
            save_part(out, part, E[[row[t] for t in part_ids[part]]], part_ids[part])
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

    df = parts["test_inputs"]
    t0 = time.perf_counter()
    save_part(out, "test_inputs", emb.encode(build_texts(df["subject"], df["body"]), show_progress=True), test_ids)
    print(f"test_inputs embedded in {time.perf_counter() - t0:.1f}s (holdout: never part of the index)")
    (out / "meta.json").write_text(json.dumps({"embedding_model": model, "prefix": prefix, "dim": emb.dim,
                                               "batch_size": emb.batch_size,
                                               "max_seq_length": cfg["embedding"]["max_seq_length"]}, indent=2))


if __name__ == "__main__":
    main()
