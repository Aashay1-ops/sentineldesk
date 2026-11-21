"""Console threshold report -> artifacts/ui/gate_report.json (read by the live console's sidebar).

Uses only the 80% split: half B (the triage calibration slice) plus hand-written probes (ui/probes.py).
Never reads the holdout. Run after scripts/train_triage.py:

    python scripts/build_ui_report.py

Writes:
  critical_curve  recall / precision / share of tickets paged at each critical threshold (half B), so the
                  sidebar slider can show the trade-off, like the prototype console did
  floors          share of half-B tickets under each confidence floor and accuracy below / above it
  burst           burst-cluster threshold: just above the similarity of *distinct* hand-written issues
  sigma           how well sigma separates hand-written in-domain tickets from off-topic ones
"""
import json

import numpy as np
from _common import REPO_ROOT, setup_logging

from pii import mask
from sentineldesk.triage.data import read_part
from sentineldesk.triage.predict import Triage
from sentineldesk.triage.text import build_text
from ui.burst import BurstDetector
from ui.examples import BURST_TICKETS
from ui.probes import IN_DOMAIN, OUT_OF_DOMAIN

QUEUE_FLOOR = 0.30   # chosen floor; the report shows the accuracy split around it
from retrieve import SIGMA  # noqa: E402  the team's "no precedent" threshold (0.86)


def main() -> None:
    setup_logging()
    tri = Triage.load(REPO_ROOT / "artifacts")
    hb = read_part(REPO_ROOT / "artifacts" / "splits", "half_b")
    preds = tri.predict_batch([mask(s) if isinstance(s, str) else "" for s in hb["subject"]],
                              [mask(b) for b in hb["body"]], hb["ticket_id"].tolist())

    y = ((hb["type"] == "Incident") & (hb["priority"] == "high")).to_numpy()
    score = np.array([p["critical"]["score"] for p in preds])
    curve = []
    for th in np.round(np.arange(0.05, 0.96, 0.01), 2):
        flag = score >= th
        curve.append({"threshold": float(th), "recall": round(float((flag & y).sum() / max(y.sum(), 1)), 4),
                      "precision": round(float((flag & y).sum() / max(flag.sum(), 1)), 4),
                      "flag_rate": round(float(flag.mean()), 4)})

    floors = {}
    for task, chosen in (("type", tri.thresholds[tri.mode]["type"]), ("queue", QUEUE_FLOOR)):
        conf = np.array([p[task]["confidence"] for p in preds])
        ok = np.array([p[task]["label"] for p in preds]) == hb[task].to_numpy()
        rows = []
        for f in np.round(np.arange(0.20, 0.71, 0.05), 2):
            lo = conf < f
            rows.append({"floor": float(f), "share_below": round(float(lo.mean()), 4),
                         "acc_below": round(float(ok[lo].mean()), 4) if lo.any() else None,
                         "acc_above": round(float(ok[~lo].mean()), 4) if (~lo).any() else None})
        lo = conf < chosen
        floors[task] = {"chosen": round(float(chosen), 4), "share_below": round(float(lo.mean()), 4),
                        "acc_below": round(float(ok[lo].mean()), 4) if lo.any() else None,
                        "acc_above": round(float(ok[~lo].mean()), 4), "grid": rows}

    enc = tri.embedder.encode
    probe_vecs = enc([build_text("", p) for p in IN_DOMAIN])
    distinct = (probe_vecs @ probe_vecs.T)[np.triu_indices(len(IN_DOMAIN), 1)]
    burst_thr = round(float(np.quantile(distinct, 0.99)) + 0.01, 4)
    bv = enc([build_text(mask(s), mask(b)) for s, b in BURST_TICKETS])
    det, alert = BurstDetector(sim_threshold=burst_thr), None   # dry-run "Simulate burst" with the real detector
    for i, (s, b) in enumerate(BURST_TICKETS):
        alert = det.add(f"B{i + 1}", bv[i], s, b, failure_like=True) or alert

    idx = np.load(REPO_ROOT / "data" / "index" / "emb.npy")
    in_best = (probe_vecs @ idx.T).max(1)
    out_best = (enc([build_text("", p) for p in OUT_OF_DOMAIN]) @ idx.T).max(1)

    report = {
        "source": "half B of the 80% split + hand-written probes (ui/probes.py); holdout never read",
        "n_half_b": len(hb), "positives": int(y.sum()),
        "critical_curve": curve, "floors": floors,
        "burst": {"threshold": burst_thr, "distinct_issue_p99": round(float(np.quantile(distinct, 0.99)), 4),
                  "demo_burst_alert_size": alert.size if alert else 0},
        "sigma": {"value": SIGMA, "in_domain_best": [round(float(x), 4) for x in in_best],
                  "off_topic_best": [round(float(x), 4) for x in out_best],
                  "in_domain_min": round(float(in_best.min()), 4),
                  "in_domain_median": round(float(np.median(in_best)), 4),
                  "off_topic_max": round(float(out_best.max()), 4),
                  "off_topic_median": round(float(np.median(out_best)), 4),
                  "in_domain_accepted": round(float((in_best >= SIGMA).mean()), 4),
                  "off_topic_rejected": round(float((out_best < SIGMA).mean()), 4)},
    }
    out = REPO_ROOT / "artifacts" / "ui"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"burst": report["burst"], "queue_floor": {k: v for k, v in floors["queue"].items() if k != "grid"},
                      "sigma": {k: v for k, v in report["sigma"].items() if not k.endswith("_best")}}, indent=2))
    print(f"wrote {out / 'gate_report.json'}")


if __name__ == "__main__":
    main()
