"""Holdout evaluation. Predictions use subject+body only; labels are loaded separately, for scoring only.
The holdout `answer` column is never loaded.

Run: python -m scripts.evaluate [--llm 20]
"""
import argparse
import json
import time

import numpy as np
from sklearn.metrics import classification_report, f1_score, recall_score

from sentineldesk import config
from sentineldesk import gate as gate_mod
from sentineldesk.data import TARGETS, load_test_inputs_only, load_test_labels, ticket_text
from sentineldesk.pipeline import detect_language
from sentineldesk.retrieval import Retriever, embed
from sentineldesk.triage import TriageModel, as_prediction, blend_with_precedents

ap = argparse.ArgumentParser()
ap.add_argument("--llm", type=int, default=0, help="tickets to run full LLM drafting on (costs API calls)")
args = ap.parse_args()

X = load_test_inputs_only()
texts = [ticket_text(s, b) for s, b in zip(X["subject"], X["body"])]
model, ret = TriageModel.load(), Retriever()

# ---------- Inference (no labels in scope) ----------
t0 = time.perf_counter()
probs = model.predict_proba(X["subject"].tolist(), X["body"].tolist())
Q = embed(texts, show_progress=True)
S, I = ret.index.search(Q, config.TOP_K)
assert set(ret.meta["ticket_id"].iloc[I.ravel()]) <= ret.kb_ids
blended = {}
for t in TARGETS:
    classes, P = probs[t]
    lab = ret.meta[t].values
    blended[t] = (classes, np.array([blend_with_precedents(classes, P[n], list(zip(lab[I[n]], S[n])))
                                     for n in range(len(X))]))
batch_ms = (time.perf_counter() - t0) * 1000 / len(X)

gate_status, lang_pred = [], []
for n, text in enumerate(texts):
    tri = {t: as_prediction(blended[t][0], blended[t][1][n]) for t in TARGETS}
    lang, lc = detect_language(text)
    lang_pred.append(lang)
    gate_status.append(gate_mod.pre_generation(text, lang, lc, tri, float(S[n, 0])).status)
gate_status = np.array(gate_status)

# ---------- Scoring ----------
Y = load_test_labels().set_index("ticket_id").loc[X["ticket_id"]]
report = {"n_test": len(X), "triage": {}}


def ece(conf, correct, bins=10):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


pred = {}
for t in TARGETS:
    y = Y[t].astype(str).values
    c0, P0 = probs[t]
    c1, P1 = blended[t]
    pred[t] = c1[P1.argmax(1)]
    majority = Y[t].value_counts().idxmax()
    report["triage"][t] = {
        "macro_f1": round(f1_score(y, pred[t], average="macro"), 4),
        "macro_f1_classifier_only": round(f1_score(y, c0[P0.argmax(1)], average="macro"), 4),
        "majority_baseline_macro_f1": round(f1_score(y, [majority] * len(y), average="macro"), 4),
        "accuracy": round(float((pred[t] == y).mean()), 4),
        "ece": round(ece(P1.max(1), pred[t] == y), 4),
    }
    print(f"\n=== {t} (blended) ===\n" + classification_report(y, pred[t], digits=3, zero_division=0))

cp, Pp = blended["priority"]
p_high = Pp[:, list(cp).index("high")]
oncall = (pred["type"] == "Incident") & (p_high >= config.GATE["high_priority_prob"])
true_hpi = ((Y["priority"] == "high") & (Y["type"] == "Incident")).values
report["high_priority_incident"] = {
    "support": int(true_hpi.sum()),
    "gate_recall": round(float(oncall[true_hpi].mean()), 4),
    "gate_precision": round(float(true_hpi[oncall].mean()), 4),
    "any_escalation_recall": round(float((gate_status[true_hpi] != gate_mod.AUTO).mean()), 4),
    "priority_high_class_recall": round(recall_score(Y["priority"], pred["priority"], labels=["high"], average="macro"), 4),
}
report["gate"] = {
    "status_share": {s: round(float((gate_status == s).mean()), 3) for s in (gate_mod.AUTO, gate_mod.REVIEW, gate_mod.ONCALL)},
    "top1_similarity_mean": round(float(S[:, 0].mean()), 3),
    "language_agreement_with_label": round(float((np.array(lang_pred) == Y["language"].values).mean()), 4),
    "note": "Dataset language label is noisy (~11% of KB rows labelled 'de' contain English text).",
}

# Single-ticket latency, as the console experiences it (pre-LLM).
from sentineldesk.pipeline import SentinelDesk  # noqa: E402

sd = SentinelDesk()
sd.burst.min_cluster = 10**9  # random sample must not trigger outage banners
lat = []
for row in X.sample(100, random_state=config.SEED).itertuples():
    lat.append(sd.process(row.subject, row.body, generate=False)["timings"]["total_ms"])
report["latency_ms"] = {"pre_llm_p50": round(float(np.percentile(lat, 50)), 1),
                        "pre_llm_p95": round(float(np.percentile(lat, 95)), 1),
                        "batch_per_ticket": round(batch_ms, 2)}

if args.llm:
    rows, llm_lat = [], []
    for row in X.sample(args.llm, random_state=7).itertuples():
        r = sd.process(row.subject, row.body)
        if r["draft"]:
            d = r["draft"]
            llm_lat.append(r["timings"]["total_ms"])
            rows.append((len(d["cited_ids"]) > 0, len(d["invalid_citations"]) > 0, d["evidence_sufficient"], d["mode"]))
    if rows:
        a = np.array([r[:3] for r in rows], dtype=float)
        report["llm_sample"] = {
            "n_drafted": len(rows), "mode": rows[0][3],
            "drafts_with_valid_citation": round(a[:, 0].mean(), 3),
            "drafts_with_hallucinated_id_removed": round(a[:, 1].mean(), 3),
            "evidence_sufficient_rate": round(a[:, 2].mean(), 3),
            "end_to_end_ms_p50": round(float(np.percentile(llm_lat, 50))),
            "end_to_end_ms_p95": round(float(np.percentile(llm_lat, 95))),
        }

out = config.ROOT / "reports"
out.mkdir(exist_ok=True)
json.dump(report, open(out / "eval_report.json", "w"), indent=2, default=float)
print(json.dumps(report, indent=2, default=float))
