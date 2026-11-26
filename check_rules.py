"""Hackathon fair-play check. Run before every push and before the demo:  python check_rules.py
Fails loudly if anything we build could get us disqualified for leakage."""
from pathlib import Path

import pandas as pd

from split import load_split

kb, test = load_split()
kb2, test2 = load_split()

# 1. One locked split: same seed -> identical ids every time.
assert kb.ticket_id.tolist() == kb2.ticket_id.tolist(), "split is not deterministic"
assert test.ticket_id.tolist() == test2.ticket_id.tolist(), "split is not deterministic"

# 2. Train and holdout never share a ticket.
assert not set(kb.ticket_id) & set(test.ticket_id), "ticket_id in both splits"

# 3. Holdout answers never leave split.py.
assert "answer" not in test.columns, "holdout answers are exposed"

# 4. The search index holds only 80%-split tickets.
idx = Path("data/index/kb.pkl")
if idx.exists():
    ids = set(pd.read_pickle(idx).ticket_id)
    assert not ids & set(test.ticket_id), "holdout ticket in the search index"
    assert ids <= set(kb.ticket_id), "index has tickets not in the 80% split"

# 5. Report (don't fail on) exact-duplicate texts across the split: they inflate metrics.
dup = test.text.isin(set(kb.text)).sum()
print(f"OK: kb={len(kb)} holdout={len(test)} index_checked={idx.exists()} "
      f"holdout texts also in kb={dup} ({dup / len(test):.1%})")
