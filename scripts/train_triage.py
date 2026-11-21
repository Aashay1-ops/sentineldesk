"""Cut train / half_a / half_b from kb BY PARAPHRASE FAMILY, then train models 0, 1, 5, build the
neighbour index (2), fit the ensemble (6) on half A, calibrate and tune thresholds on half B.
Never touches holdout labels."""
import json
import time

import numpy as np
from _common import parser, setup_logging

from sentineldesk.triage.config import artifacts_dir, load_config, model_slug
from sentineldesk.triage.data import family_calib_split, read_part, write_calib_parts
from sentineldesk.triage.training import train_all


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--model", default=None, help="embedding model whose saved embeddings to use")
    p.add_argument("--fallback-only", action="store_true", help="only model 0 + critical detector (M3)")
    args = p.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    root = artifacts_dir(cfg, args.artifacts, args.sample)
    model = args.model or cfg["embedding"]["model"]
    edir = root / "embeddings" / model_slug(model)
    if not (edir / "meta.json").exists() or not (edir / "kb.npy").exists():
        have = sorted(p.name for p in (root / "embeddings").glob("*")) if (root / "embeddings").exists() else []
        raise SystemExit(f"No kb embeddings for {model} in {edir}.\n"
                         f"Run first:  python scripts/embed.py --model {model}"
                         + (" --sample " + str(args.sample) if args.sample else "")
                         + f"\nEmbeddings that exist: {have or 'none'}")

    splits = root / "splits"
    kb = read_part(splits, "kb")
    ids = json.loads((edir / "kb_ids.json").read_text())
    if ids != kb["ticket_id"].tolist():
        raise ValueError("kb embedding order does not match splits/kb.parquet; re-run scripts/embed.py")
    E = np.load(edir / "kb.npy")
    threshold = cfg["calibration"].get("family_threshold", 0.98)
    pos, fam = family_calib_split(kb, E, threshold)
    write_calib_parts(kb, pos, fam, splits)
    sizes = {k: len(v) for k, v in pos.items()}
    print(f"family-based split (cosine >= {threshold}): {sizes}, "
          f"{int((np.bincount(fam) > 1).sum())} families with reworded copies, 0 shared across parts")

    parts = {k: kb.iloc[v].reset_index(drop=True) for k, v in pos.items()}
    embs = {k: E[v] for k, v in pos.items()}
    test_ids = read_part(splits, "test_inputs")["ticket_id"].tolist()  # ids only, for the leakage assert
    embed_meta = json.loads((edir / "meta.json").read_text())
    t0 = time.perf_counter()
    summary = train_all(cfg, parts, embs, root / "triage", embed_meta, test_ids=test_ids,
                        fallback_only=args.fallback_only)
    print(json.dumps({"times_s": summary["times"], "total_s": round(time.perf_counter() - t0, 1),
                      "thresholds": summary["thresholds"],
                      "half_b (new tickets: no reworded copy in train)":
                          {m: {t: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                               for t, r in d.items()} for m, d in summary["calibration"].items()}}, indent=2))


if __name__ == "__main__":
    main()
