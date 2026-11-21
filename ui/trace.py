"""Full trace of one ticket: the actual data going in and out of every pipeline stage.

Display only: everything shown was computed by the real pipeline (Desk.analyse / Desk.decide). Which gate
rules fired is read from the gate's own decision, never re-derived here; the compared values are context.
"""
from __future__ import annotations

import json


def gate_rows(a: dict, gate, d, cfg) -> list[dict]:
    tri, fired = a["triage"], {r.code for r in gate.reasons}
    best = max((p["similarity"] for p in a["precedents"]), default=0.0)
    row = lambda rule, value, test, code, effect: {
        "rule": rule, "value": value, "test": test, "result": f"✗ fired → {effect}" if code in fired else "✓ pass"}
    return [
        row("Critical-incident detector", f"{tri['critical']['score']:.3f}", f"< {cfg.critical_threshold:.3f}",
            "CRITICAL_SCORE", "page on-call"),
        row("Type confidence", f"{tri['type']['confidence']:.3f}", f"≥ {cfg.min_type_confidence:.3f}",
            "LOW_CONFIDENCE" if tri["type"]["confidence"] < cfg.min_type_confidence else "-", "review"),
        row("Queue confidence", f"{tri['queue']['confidence']:.3f}", f"≥ {cfg.min_queue_confidence:.3f}",
            "LOW_CONFIDENCE" if tri["queue"]["confidence"] < cfg.min_queue_confidence else "-", "review"),
        row("Best precedent similarity", f"{best:.3f}", f"≥ σ {cfg.min_similarity:.3f}", "NO_PRECEDENT", "review"),
        row("Sensitive request (refund, SLA, legal …)", ", ".join(a["sensitive_hits"]) or "none", "none",
            "SENSITIVE_REQUEST", "review"),
        row("Prompt injection", ", ".join(a["injection_hits"]) or "none", "none", "PROMPT_INJECTION", "review"),
        row("Draft verification", f"{len(d.issues)} issue(s)", "0", "DRAFT_GUARD", "review, draft hidden"),
        row("LLM unsupported asks", ", ".join(d.unsupported_requests) or "none", "none", "LLM_FLAGGED", "review"),
    ]


def render(st, a: dict, gate, d, cfg, bars, queue_item: dict | None) -> None:
    import pandas as pd

    st.markdown("### 🔬 Full trace: one ticket, input → output")
    st.caption(f"{a['ref']} · every value below is the real intermediate output of this run. "
               "Open the stages one by one while presenting.")
    open_all = st.toggle("Expand all stages", value=False, key=f"trace_all_{a['ref']}")
    t = a["timings_ms"]
    ms = lambda k: f" · {t[k]:.0f} ms" if t.get(k, 0) >= 1 else (" · <1 ms" if k in t else "")

    with st.expander("① Input: raw ticket as received", expanded=True):
        st.text_input("Subject", a["subject"] or "(none)", disabled=True, key=f"tr_s_{a['ref']}")
        st.text_area("Body", a["body"], disabled=True, height=110, key=f"tr_b_{a['ref']}")

    with st.expander(f"② PII mask{ms('mask')}", expanded=open_all):
        c1, c2 = st.columns(2)
        c1.markdown("**Before**")
        c1.code(f"{a['subject']}\n{a['body']}".strip(), language=None, wrap_lines=True)
        c2.markdown("**After** (all later stages and the LLM only see this)")
        c2.code(a["masked_text"].strip(), language=None, wrap_lines=True)
        st.caption("Replaced: " + (", ".join(f"{k} ×{v}" for k, v in a["pii_counts"].items()) or "nothing (no PII found)"))

    with st.expander(f"③ Language detection{ms('language')}", expanded=open_all):
        st.json({"language": a["language"], "confidence": a["language_confidence"],
                 "used for": "reply language, verify's language check, cross-lingual retrieval"})

    with st.expander(f"④ Guardrails{ms('guardrails')}", expanded=open_all):
        st.json({"prompt_injection_matches": a["injection_hits"] or [],
                 "sensitive_request_matches": a["sensitive_hits"] or {}})

    tri = a["triage"]
    with st.expander(f"⑤ Triage (calibrated ensemble){ms('triage')}", expanded=open_all):
        cols = st.columns(3)
        for col, head in zip(cols, ("type", "queue", "priority")):
            col.markdown(f"**{head.title()}** → `{tri[head]['label']}` ({tri[head]['confidence']:.0%})")
            col.markdown(bars(tri[head]["probs"]), unsafe_allow_html=True)
        cr = tri["critical"]
        st.markdown(f"**Critical detector:** score `{cr['score']:.3f}` vs threshold `{cr['threshold']:.3f}` → "
                    f"{'🚨 fires' if cr['fires'] else 'quiet'}"
                    + (f" · outage keywords: {', '.join(cr['keyword_hits'])}" if cr.get("keyword_hits") else ""))
        expl = (tri.get("neighbours") or {}).get("explanation") or {}
        if expl:
            st.caption("Neighbour votes (similar training tickets): " + " · ".join(str(v) for v in expl.values()))

    with st.expander(f"⑥ Retrieval: hybrid BM25 + dense, RRF{ms('retrieve')}", expanded=open_all):
        st.markdown("**Query** (masked ticket text)")
        st.code(a["masked_text"][:600], language=None, wrap_lines=True)
        st.dataframe(pd.DataFrame([{
            "ticket_id": p["ticket_id"], "similarity": round(p["similarity"], 3), "lang": p.get("language"),
            "type / queue": f"{p.get('type')} / {p.get('queue')}", "subject": (p.get("subject") or "")[:60],
            "answer (start)": (p.get("answer") or "")[:140].replace("\n", " "),
        } for p in a["precedents"]]), hide_index=True, width="stretch")

    with st.expander(f"⑦ Escalation gate: every rule{ms('gate')}", expanded=open_all):
        st.dataframe(pd.DataFrame(gate_rows(a, gate, d, cfg)), hide_index=True, width="stretch")
        st.markdown(f"**Before drafting:** `{a.get('gate_pre', gate.status.value)}` · **after verify:** "
                    f"`{gate.status.value}`")

    tr = d.trace or {}
    with st.expander(f"⑧ Draft{ms('draft')}", expanded=open_all):
        if tr.get("prompt"):
            system, user = tr["prompt"]
            st.markdown("**Prompt sent to the LLM** (system instruction + user content)")
            st.code(system, language=None, wrap_lines=True)
            st.code(user[:4000], language=None, wrap_lines=True)
        if tr.get("raw"):
            st.markdown(f"**Raw LLM response** ({tr['raw'].get('model')}, {tr['raw'].get('latency_ms')} ms)")
            st.code(json.dumps({k: tr["raw"][k] for k in ("reply", "cited_ids", "unsupported_asks") if k in tr["raw"]},
                               ensure_ascii=False, indent=2), language="json", wrap_lines=True)
        elif tr.get("error"):
            st.warning(f"LLM call failed: {tr['error'][:300]} → offline extractive draft used.")
        else:
            st.info({"holding": "No LLM call: holding reply (ticket goes to a human; nothing generated from this text).",
                     "extractive": "No LLM call: the gate did not clear this ticket (or the LLM is off), so an offline "
                                   "extractive draft from the closest precedent is prepared for the agent."}
                    .get(d.mode, "No LLM call."))
        st.caption(d.internal_note)

    with st.expander(f"⑨ Verify{ms('verify')}", expanded=open_all):
        if tr.get("checks"):
            st.dataframe(pd.DataFrame([{"check": c["check"], "result": "✓ pass" if c["passed"] else "✗ fail",
                                        "evidence": str(c["evidence"])[:200]} for c in tr["checks"]]),
                         hide_index=True, width="stretch")
            st.caption("sentineldesk/verify.py: a draft is shown only if every check passes.")
        else:
            st.markdown("Offline rule checks (ui/drafting.py): " +
                        ("✗ " + "; ".join(d.issues) if d.issues else "✓ citations, figures, commitments, PII and "
                                                                      "language all pass"))

    with st.expander("⑩ Output", expanded=True):
        st.markdown(f"**Decision:** `{gate.status.value}` · reasons: "
                    + (", ".join(f"`{r.code}`" for r in gate.reasons) or "none"))
        st.markdown(f"**Reply** ({d.mode}, cites {', '.join(d.cited_ticket_ids) or 'none'})")
        st.code(d.reply, language=None, wrap_lines=True)
        if queue_item:
            st.markdown(f"**Review queue:** {queue_item['ref']} is `{queue_item['state']}` "
                        f"({'on-call first' if queue_item['status'] == 'ESCALATE_ON_CALL' else 'review'})")
        else:
            st.markdown("**Review queue:** not queued (auto-draft ready, no human needed)")
        st.caption(f"End-to-end: {t.get('total', 0) + sum(t.get(k, 0) for k in ('gate', 'draft', 'verify')):.0f} ms")


if __name__ == "__main__":  # self-check: gate rows mirror the gate's own decision
    from types import SimpleNamespace as NS

    from ui import mocks
    from ui.drafting import DraftResult
    from ui.gate import GateConfig, evaluate_gate

    cfg = GateConfig()
    tri = mocks.mock_triage("VPN down", "VPN gateway down since 8am, all users offline")
    a = {"triage": tri, "precedents": [{"similarity": 0.5}], "sensitive_hits": {}, "injection_hits": []}
    g = evaluate_gate(tri, a["precedents"], {}, [], cfg)
    rows = gate_rows(a, g, DraftResult(reply="x", mode="holding", cited_ticket_ids=[]), cfg)
    sim = next(r for r in rows if r["rule"].startswith("Best precedent"))
    assert sim["result"].startswith("✗") and "NO_PRECEDENT" in {r.code for r in g.reasons}
    assert all(r["result"] == "✓ pass" for r in rows if "injection" in r["rule"].lower())
    print("trace self-check OK")
