"""Train models 0, 1, 5; build the neighbour index (2); fit the ensemble (6) on half A;
calibrate and tune thresholds on half B. Never touches holdout labels.

train / half_a / half_b come from scripts/embed.py, which redraws them BY PARAPHRASE FAMILY
(data audit #3/#4): no half-A / half-B ticket has a reworded copy in train."""
import json
import time

import numpy as np
from _common import parser, setup_logging

from sentineldesk.triage.config import artifacts_dir, load_config, model_slug
from sentineldesk.triage.data import read_part
from sentineldesk.triage.training import train_all


def load_emb(edir, part, df):
    ids = json.loads((edir / f"{part}_ids.json").read_text())
    if ids != df["ticket_id"].tolist():
        raise ValueError(f"Embedding order for {part} does not match the split; re-run scripts/embed.py")
    return np.load(edir / f"{part}.npy")


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
    if not (edir / "meta.json").exists():
        have = sorted(p.name for p in (root / "embeddings").glob("*")) if (root / "embeddings").exists() else []
        raise SystemExit(f"No embeddings for {model} in {edir}.\n"
                         f"Run first:  python scripts/embed.py --model {model}"
                         + (" --sample " + str(args.sample) if args.sample else "")
                         + f"\nEmbeddings that exist: {have or 'none'}")
    parts = {k: read_part(root / "splits", k) for k in ("train", "half_a", "half_b")}
    embs = {k: load_emb(edir, k, parts[k]) for k in parts}
    test_ids = read_part(root / "splits", "test_inputs")["ticket_id"].tolist()  # ids only, for the leakage assert
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
