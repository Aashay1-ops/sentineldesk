"""The one locked 80/20 split everyone imports. Data owner: replace the source file
here if we switch to main + deduped 20k; everything downstream keeps working."""
import pandas as pd


def load_split(path="data/tickets.csv", seed=42, frac=0.8):
    df = pd.read_csv(path).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    # Data audit #6: 1,233 bodies / 1,005 answers contain the two characters backslash + n, and
    # 640 / 176 contain HTML <br>; make both real line breaks. sentineldesk/triage/text.py:_clean
    # does the same, in the same order, for live tickets (texts must stay identical).
    for c in ("subject", "body", "answer"):
        df[c] = df[c].str.replace("\\n", "\n", regex=False).str.replace(r"(?i)<br\s*/?>", "\n", regex=True)
    df["text"] = df["subject"].fillna("") + "\n" + df["body"].fillna("")
    cut = int(len(df) * frac)
    kb, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    # Leakage guard: holdout answers never leave this function.
    return kb, test.drop(columns=["answer"])
