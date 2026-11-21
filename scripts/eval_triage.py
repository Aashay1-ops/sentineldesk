"""Holdout evaluation. Order matters for leakage:
  1. predict on test_inputs and SAVE predictions,
  2. only then read test_labels and score.
Never feed these numbers back into training, keywords or thresholds."""
import json
import time

import numpy as np
import pandas as pd
from _common import parser, setup_logging

from sentineldesk.triage import TASKS
from sentineldesk.triage.calibration import ece, reliability
from sentineldesk.triage.config import artifacts_dir, load_config, model_slug, results_dir
from sentineldesk.triage.data import assert_no_test_ids, read_part, read_test_labels
from sentineldesk.triage.labels import encode
from sentineldesk.triage.predict import Triage
from sentineldesk.triage.text import build_texts

# Validated default palette (dataviz skill, light mode): categorical slots 1-2 + blue sequential ramp.
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3dc"
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
VARIANTS = {"m0": "model 0 (embeddings)", "m1": "model 1 (TF-IDF)", "m2": "model 2 (neighbours)",
            "fallback": "fallback (0 + temperature)", "full": "full ensemble (0+1+2+6)"}


def predict_holdout(triage: Triage, test: pd.DataFrame, chunk: int = 512) -> tuple[pd.DataFrame, dict]:
    texts = build_texts(test["subject"], test["body"])
    ids = test["ticket_id"].tolist()
    store: dict[str, list] = {}
    for s in range(0, len(texts), chunk):
        a = triage.predict_arrays(texts[s:s + chunk], ids[s:s + chunk])
        probs = {"m0": a["p0"], "fallback": a["fallback"], "full": a["final"]}
        if "p1" in a:
            probs.update(m1=a["p1"], m2=a["p2"], ens_raw=a["ens_raw"])
        elif "p2" in a:
            probs["m2"] = a["p2"]
        for v, d in probs.items():
            for t in TASKS:
                store.setdefault(f"probs_{v}_{t}", []).extend(d[t].tolist())
        store.setdefault("critical_score", []).extend(a["critical_score"].tolist())
        store.setdefault("keyword_hits", []).extend(a["keyword_hits"])
    pred = pd.DataFrame({"ticket_id": ids, **store})
    thr = triage.thresholds["critical"]["threshold"]
    pred["critical_fires"] = pred["critical_score"] >= thr
    for t in TASKS:
        for v in VARIANTS:
            col = f"probs_{v}_{t}"
            if col in pred:
                pred[f"pred_{v}_{t}"] = [triage.labels[t][int(np.argmax(p))] for p in pred[col]]
    return pred, {"critical_threshold": thr}


def copy_flags(root, triage: Triage, test: pd.DataFrame, threshold: float) -> np.ndarray | None:
    """True where a holdout ticket has a reworded copy in kb (nearest-kb cosine >= threshold).
    Text-only (embeddings of the holdout inputs), so no holdout label or answer is read."""
    edir = root / "embeddings" / model_slug(triage.meta["embedding_model"])
    need = [edir / f for f in ("kb.npy", "test_inputs.npy", "test_inputs_ids.json")]
    if not all(f.exists() for f in need):
        return None
    if json.loads((edir / "test_inputs_ids.json").read_text()) != test["ticket_id"].tolist():
        return None
    kb, te = np.load(edir / "kb.npy"), np.load(edir / "test_inputs.npy")
    best = np.concatenate([(te[s:s + 1024] @ kb.T).max(1) for s in range(0, len(te), 1024)])
    return best >= threshold


def latency(triage: Triage, test: pd.DataFrame, n: int = 200) -> dict:
    rows = test.head(n)
    triage.predict(rows.iloc[0]["subject"], rows.iloc[0]["body"])  # warm-up
    ms = []
    for s, b in zip(rows["subject"], rows["body"]):
        t0 = time.perf_counter()
        triage.predict(s, b)
        ms.append((time.perf_counter() - t0) * 1000)
    return {"n": len(ms), "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95))}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_confusion(y_true, y_pred, labels, title, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n, 0.8 + 0.65 * n))
    im = ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP), vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted", color=INK)
    ax.set_ylabel("true (row-normalized)", color=INK)
    ax.set_title(title, color=INK, loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(before, after, y, task, bins, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect calibration")
    for probs, color, name in ((before, ORANGE, "before temperature"), (after, BLUE, "after temperature")):
        r = reliability(probs, y, bins)
        xs = [c for c, k in zip(r["conf"], r["count"]) if k]
        ys = [a for a, k in zip(r["acc"], r["count"]) if k]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.5, label=f"{name} (ECE {ece(probs, y, bins):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence", color=INK)
    ax.set_ylabel("accuracy", color=INK)
    ax.set_title(f"Reliability: {task} (holdout)", color=INK, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(y, scores, thr, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    prec, rec, thrs = precision_recall_curve(y, scores)
    fires = scores >= thr
    r_at = (fires & (y == 1)).sum() / max(y.sum(), 1)
    p_at = (fires & (y == 1)).sum() / max(fires.sum(), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    _style(ax)
    ax.plot(rec, prec, color=BLUE, linewidth=2)
    ax.plot([r_at], [p_at], marker="o", markersize=9, color=BLUE, markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"threshold {thr:.2f} (tuned on half B)\nrecall {r_at:.2f}, precision {p_at:.2f}",
                (r_at, p_at), textcoords="offset points", xytext=(-150, -40), fontsize=8, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall", color=INK)
    ax.set_ylabel("precision", color=INK)
    ax.set_title("Critical detector: high-priority Incidents (holdout)", color=INK, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--no-latency", action="store_true")
    args = p.parse_args()
    setup_logging()
    from sklearn.metrics import classification_report, f1_score

    cfg = load_config(args.config)
    root = artifacts_dir(cfg, args.artifacts, args.sample)
    out = results_dir(cfg, args.sample)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    splits = root / "splits"
    test = read_part(splits, "test_inputs")
    triage = Triage.load(root)
    assert_no_test_ids(triage.nn.ids if triage.nn is not None else [], test["ticket_id"])
    for part in ("train", "half_a", "half_b"):
        assert_no_test_ids(read_part(splits, part)["ticket_id"], test["ticket_id"])

    # 1) predict and save BEFORE any label is read
    t0 = time.perf_counter()
    pred, info = predict_holdout(triage, test)
    pred_path = out / "test_predictions.parquet"
    pred.to_parquet(pred_path, index=False)
    predict_s = time.perf_counter() - t0

    # 2) only now unlock labels
    gold = read_test_labels(splits, pred_path).set_index("ticket_id").loc[pred["ticket_id"]].reset_index()
    metrics: dict = {"mode": triage.mode, "n_test": len(pred), "predict_seconds": round(predict_s, 1),
                     "embedding_model": triage.meta["embedding_model"], "macro_f1": {}, "per_class_full": {},
                     "calibration_holdout": {}}
    bins = cfg["calibration"]["ece_bins"]
    best = "full" if triage.mode == "full" else "fallback"
    for t in TASKS:
        metrics["macro_f1"][t] = {VARIANTS[v]: round(f1_score(gold[t], pred[f"pred_{v}_{t}"], average="macro"), 4)
                                  for v in VARIANTS if f"pred_{v}_{t}" in pred}
        metrics["per_class_full"][t] = classification_report(gold[t], pred[f"pred_{best}_{t}"], output_dict=True,
                                                             zero_division=0)
        y = encode(gold[t], triage.labels[t])
        after = np.array(pred[f"probs_{best}_{t}"].tolist())
        before = np.array(pred[f"probs_{'ens_raw' if best == 'full' else 'm0'}_{t}"].tolist())
        metrics["calibration_holdout"][t] = {"ece_before": ece(before, y, bins), "ece_after": ece(after, y, bins)}
        plot_reliability(before, after, y, t, bins, out / "plots" / f"reliability_{t}.png")
        if t in ("priority", "queue"):
            plot_confusion(gold[t], pred[f"pred_{best}_{t}"], triage.labels[t],
                           f"{t}: {VARIANTS[best]} (holdout)", out / "plots" / f"confusion_{t}.png")

    crit_true = ((gold["type"] == "Incident") & (gold["priority"] == "high")).to_numpy()

    # All / known (reworded copy in kb) / novel holdout tickets (docs/DATA_PREPROCESSING.md item 2)
    known = copy_flags(root, triage, test, cfg["calibration"].get("family_threshold", 0.98))
    if known is not None:
        metrics["slices"] = {}
        for name, m in (("all", np.ones(len(pred), bool)), ("known_copy_in_kb", known), ("novel", ~known)):
            if not m.any():
                continue
            sl = {"n": int(m.sum())}
            for t in TASKS:
                sl[f"macro_f1_{t}"] = round(f1_score(gold[t][m], pred[f"pred_{best}_{t}"][m], average="macro"), 4)
            ct, fires = crit_true[m], pred["critical_fires"].to_numpy()[m]
            sl["critical_recall"] = round(float((fires & ct).sum() / max(ct.sum(), 1)), 4)
            metrics["slices"][name] = sl

    def pr(flag):
        tp = int((flag & crit_true).sum())
        return {"recall": tp / max(int(crit_true.sum()), 1), "precision": tp / max(int(flag.sum()), 1),
                "flagged": int(flag.sum())}

    labels_only = ((pred[f"pred_{best}_type"] == "Incident") & (pred[f"pred_{best}_priority"] == "high")).to_numpy()
    metrics["high_priority_incident"] = {
        "positives": int(crit_true.sum()),
        "labels_only": pr(labels_only),
        "critical_detector": pr(pred["critical_fires"].to_numpy()),
        "labels_or_detector": pr(labels_only | pred["critical_fires"].to_numpy()),
        **info,
    }
    plot_pr(crit_true.astype(int), pred["critical_score"].to_numpy(), info["critical_threshold"],
            out / "plots" / "critical_pr_curve.png")
    if not args.no_latency:
        metrics["latency_single_ticket"] = latency(triage, test)
    (out / "triage_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("mode", "macro_f1", "high_priority_incident", "calibration_holdout")
                      + (("slices",) if "slices" in metrics else ())
                      + (("latency_single_ticket",) if "latency_single_ticket" in metrics else ())}, indent=2))
    print(f"wrote {out / 'triage_metrics.json'} and plots in {out / 'plots'}")


if __name__ == "__main__":
    main()
