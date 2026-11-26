"""Helper: mine candidate outage keywords from `train` tags (Outage / Disruption). Output is for a
HUMAN to review in keywords/candidates.txt; it is never added to outage_keywords.yaml automatically."""
import re
from collections import Counter

import numpy as np
from _common import REPO_ROOT, parser, setup_logging

from sentineldesk.triage.config import artifacts_dir, load_config
from sentineldesk.triage.data import read_part
from sentineldesk.triage.text import build_texts

TARGET_TAGS = {"Outage", "Disruption"}
TOKEN = re.compile(r"\w+", re.UNICODE)


def grams(text):
    toks = TOKEN.findall(text.lower())
    return set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--top", type=int, default=150)
    p.add_argument("--min-count", type=int, default=20)
    args = p.parse_args()
    setup_logging()
    train = read_part(artifacts_dir(load_config(args.config), args.artifacts, args.sample) / "splits", "train")
    tag_cols = [c for c in train.columns if c.startswith("tag_")]
    all_tags = Counter(v for c in tag_cols for v in train[c].dropna())
    found = {t: all_tags.get(t, 0) for t in TARGET_TAGS}
    print("target tag counts in train:", found)
    if not any(found.values()):
        raise SystemExit("No Outage/Disruption tags found in train; nothing to mine")
    pos = train[tag_cols].isin(TARGET_TAGS).any(axis=1).to_numpy()
    texts = build_texts(train["subject"], train["body"])
    cp, cn = Counter(), Counter()
    for t, is_pos in zip(texts, pos):
        (cp if is_pos else cn).update(grams(t))
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    rows = []
    for g, c in cp.items():
        if c < args.min_count:
            continue
        lift = (c / n_pos) / ((cn[g] + 1) / (n_neg + 1))
        rows.append((np.log(lift) * np.log1p(c), lift, c, cn[g], g))
    rows.sort(reverse=True)
    out = REPO_ROOT / "keywords" / "candidates.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# mined from train only: {n_pos} Outage/Disruption-tagged vs {n_neg} other tickets\n")
        f.write("# lift\tpos_count\tneg_count\tterm   (REVIEW BY HAND; do not paste blindly)\n")
        for _, lift, c, n, g in rows[: args.top]:
            f.write(f"{lift:.2f}\t{c}\t{n}\t{g}\n")
    print(f"wrote {min(len(rows), args.top)} candidates to {out}")


if __name__ == "__main__":
    main()
