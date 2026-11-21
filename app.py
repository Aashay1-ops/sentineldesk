"""SentinelDesk - Live Triage Console (Streamlit). Same console as the prototype, on the team's modules.

    streamlit run app.py                          # real triage + index where built, stand-ins otherwise
    SENTINELDESK_UI_MOCK=1 streamlit run app.py   # mocks only (no models needed)
    http://localhost:8501/?example=vpn_outage     # preload and run a demo ticket (slugs in ui/examples.py)

Build first: scripts/build_split.py -> embed.py -> train_triage.py (triage + data/index), then
scripts/build_ui_report.py (sidebar threshold captions + burst threshold, from half B of the 80% split).
"""
from __future__ import annotations

import html
import re
import time
from dataclasses import replace
from datetime import datetime

import pandas as pd
import streamlit as st

from ui import flow, hitl, trace
from ui.drafting import clean_resolution
from ui.engine import Desk
from ui.examples import BURST_TICKETS, EXAMPLES, STREAM_TICKETS
from ui.gate import Status

st.set_page_config(page_title="SentinelDesk", page_icon="🛰️", layout="wide")

# ----------------------------------------------------------------------------- styles
STATUS_STYLE = {
    Status.AUTO: ("✅", "Auto-Draft Ready", "#0ca30c", "rgba(12,163,12,0.08)"),
    Status.REVIEW: ("⚠️", "Human Review Required", "#b97f00", "rgba(250,178,25,0.14)"),
    Status.ONCALL: ("🚨", "Escalate to On-Call Engineer", "#d03b3b", "rgba(208,59,59,0.09)"),
}
st.markdown(
    """
<style>
:root { --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#7a7974; --line:#e4e2dc; --card:#ffffff; --blue:#2a78d6; --blue-soft:#cde2fb; }
.block-container { padding-top: 3.6rem; max-width: 1400px; }
.sd-title { font-size: 1.55rem; font-weight: 700; color: var(--ink); margin: 0; }
.sd-sub { color: var(--ink-2); font-size: .92rem; margin-top: .1rem; }
.sd-badge { border-radius: 10px; padding: .85rem 1rem; border: 1px solid; margin-bottom: .6rem; }
.sd-badge h3 { margin: 0 0 .35rem 0; font-size: 1.15rem; }
.sd-reason { font-size: .86rem; color: var(--ink); margin: .2rem 0; }
.sd-code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .74rem; padding: 1px 6px;
           border-radius: 5px; background: #f0efec; color: var(--ink-2); margin-right: .35rem; }
.sd-card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: .8rem .95rem; margin-bottom: .6rem; }
.sd-k { color: var(--ink-3); font-size: .74rem; text-transform: uppercase; letter-spacing: .04em; }
.sd-v { color: var(--ink); font-size: 1.05rem; font-weight: 650; }
.sd-conf { color: var(--ink-2); font-size: .8rem; }
.sd-draft { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.1rem;
            line-height: 1.5; color: var(--ink); font-size: .95rem; }
.sd-cite { background: var(--blue-soft); color: #104281; border-radius: 5px; padding: 0 5px; font-family: ui-monospace, Menlo, monospace;
           font-size: .8rem; white-space: nowrap; }
.sd-bar-row { display: grid; grid-template-columns: 175px 1fr 48px; align-items: center; gap: .5rem; margin: 3px 0; font-size: .82rem; color: var(--ink); }
.sd-bar-track { background: #f0efec; border-radius: 4px; height: 10px; overflow: hidden; }
.sd-bar-fill { height: 10px; border-radius: 0 4px 4px 0; }
.sd-num { text-align: right; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.sd-banner { background: rgba(208,59,59,0.10); border: 1px solid #d03b3b; border-left: 6px solid #d03b3b; border-radius: 10px;
             padding: .8rem 1rem; margin-bottom: .9rem; color: var(--ink); }
.sd-banner b { color: #a32626; }
.sd-chip { display:inline-block; border:1px solid var(--line); border-radius: 999px; padding: 1px 8px; font-size: .75rem; color: var(--ink-2); margin: 2px 4px 2px 0; }
.sd-prec-head { display:flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.sd-sim { font-variant-numeric: tabular-nums; color: var(--ink-2); font-size: .85rem; }
.sd-src { display:flex; justify-content: space-between; gap: .5rem; font-size: .8rem; color: var(--ink-2); margin: 2px 0; }
@media (max-width: 700px) { .sd-bar-row { grid-template-columns: 110px 1fr 42px; } }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading triage models, past-ticket index and embedder ...")
def get_desk() -> Desk:
    return Desk.load()


desk = get_desk()
report = desk.report
ss = st.session_state
for key, val in (("history", []), ("queue", []), ("analysis", None), ("subject", ""), ("body", ""),
                 ("_pending_run", False)):
    ss.setdefault(key, val)
st.markdown(flow.CSS, unsafe_allow_html=True)

PLACEHOLDER = "— choose an example ticket —"
LABELS = {label: slug for slug, (label, _, _) in EXAMPLES.items()}


def record(a: dict, gate, draft) -> None:
    hitl.enqueue(ss.queue, a, gate, draft)  # REVIEW / ON-CALL tickets go to the human review queue
    tri = a["triage"]
    ss.history.insert(0, {
        "ref": a["ref"], "time": datetime.now().strftime("%H:%M:%S"), "subject": a["subject"] or "(no subject)",
        "status": gate.status.value, "type": tri["type"]["label"], "queue": tri["queue"]["label"],
        "priority": tri["priority"]["label"], "lang": a["language"],
        "best_sim": a["precedents"][0]["similarity"] if a["precedents"] else None,
        "latency_ms": a["timings_ms"]["total"],
    })


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Pipeline settings")
    cross_lingual = st.toggle("Cross-lingual precedent search", value=True,
                              help="Let German tickets use English resolutions (and vice versa); the reply stays in the requester's language.")
    top_k = st.slider("Precedents to retrieve (k)", 3, 5, desk.cfg.top_k)
    stage_pause = st.slider("Stage animation (s per stage)", 0.0, 1.0, 0.35, 0.05,
                            help="Pause between pipeline stages in the live strip, so the audience can follow. "
                                 "The work itself runs at full speed; 0 = no animation.")

    with st.expander("Gate thresholds (tuned on half B)"):
        curve = {round(c["threshold"], 2): c for c in report.get("critical_curve", [])}
        crit = st.slider("Critical-incident score", 0.05, 0.95, round(desk.cfg.critical_threshold, 2), 0.01)
        c = curve.get(round(crit, 2))
        if c:
            st.caption(f"Half B: on-call recall {c['recall']:.0%}, precision {c['precision']:.0%}, "
                       f"{c['flag_rate']:.0%} of tickets paged.")
        type_floor = st.slider("Min type confidence", 0.0, 0.9, round(desk.cfg.min_type_confidence, 2), 0.01)
        queue_floor = st.slider("Min queue confidence", 0.0, 0.9, round(desk.cfg.min_queue_confidence, 2), 0.01)
        qf = report.get("floors", {}).get("queue", {}).get("grid", [])
        row = min(qf, key=lambda r: abs(r["floor"] - queue_floor)) if qf else None
        if row and abs(row["floor"] - queue_floor) < 0.026 and row["acc_below"] is not None:
            st.caption(f"Half B: {row['share_below']:.0%} of tickets below {row['floor']:.2f}; queue accuracy "
                       f"{row['acc_below']:.0%} below vs {row['acc_above']:.0%} above.")
        sigma = st.slider("Min precedent similarity (σ)", 0.75, 0.97, float(desk.cfg.min_similarity), 0.0025)
        sg = report.get("sigma", {})
        if sg.get("in_domain_best"):
            acc = sum(x >= sigma for x in sg["in_domain_best"]) / len(sg["in_domain_best"])
            rej = sum(x < sigma for x in sg["off_topic_best"]) / len(sg["off_topic_best"])
            st.caption(f"Hand-written probes: {acc:.0%} of real tickets have a precedent, {rej:.0%} of off-topic "
                       "ones are rejected.")
    cfg = replace(desk.cfg, critical_threshold=crit, min_type_confidence=type_floor,
                  min_queue_confidence=queue_floor, min_similarity=sigma, top_k=top_k)

    m = desk.metrics()
    st.markdown(f"### 📈 {m.get('title', 'Metrics')}")
    if m:
        f1 = m["macro_f1"]
        st.markdown(
            f"""
| | Macro-F1 |
|---|---|
| Type | **{f1['type']:.3f}** |
| Queue | **{f1['queue']:.3f}** |
| Priority | **{f1['priority']:.3f}** |
| High-prio incident recall | **{m['crit_recall']:.3f}** |
"""
        )
        if m["n"]:
            st.caption(f"{m['n']:,} unseen holdout tickets (scripts/eval_triage.py).")
        else:
            st.caption(f"Half B of the 80% split ({report.get('n_half_b', '~1,100')} tickets). The holdout is "
                       "scored once, at the end (scripts/eval_triage.py).")
    else:
        st.caption("Train triage (scripts/train_triage.py) to populate.")

    st.markdown("### 📡 Outage detection")
    st.caption(f"Window {int(desk.burst.window / 60)} min · cluster ≥ {desk.burst.min_cluster} similar failure tickets "
               f"(cos ≥ {desk.burst.sim_threshold:.3f})")
    b1, b2 = st.columns(2)
    if b1.button("Simulate burst", width="stretch"):
        for subj, body in BURST_TICKETS:
            record(*desk.process(subj, body, cfg, k=top_k, cross_lingual=cross_lingual))
        st.toast(f"Injected {len(BURST_TICKETS)} SSO-failure tickets into the live window")
    if b2.button("Reset window", width="stretch"):
        desk.burst.reset()
        st.toast("Burst window cleared")

    with st.expander("Pipeline components"):
        st.markdown("".join(f'<div class="sd-src"><span>{html.escape(n)}</span><span>{html.escape(s)}</span></div>'
                            for n, s in desk.sources.items()), unsafe_allow_html=True)
        st.caption("real = team module · stand-in = in ui/ until the team's module lands")

# ----------------------------------------------------------------------------- header + banner
st.markdown('<p class="sd-title">🛰️ SentinelDesk · Incident Triage Console</p>'
            '<p class="sd-sub">Multilingual triage, escalation gate and precedent-grounded reply drafting (EN/DE)</p>',
            unsafe_allow_html=True)

alert = desk.burst.active
if alert and time.time() - alert.last_seen <= desk.burst.window:
    kw = ", ".join(alert.keywords) or "similar failure reports"
    subj = " · ".join(html.escape(s) for s in alert.sample_subjects)
    st.markdown(
        f'<div class="sd-banner">🚨 <b>MAJOR INCIDENT - possible outage in progress.</b> {alert.size} semantically similar failure '
        f'tickets in the last {int((alert.last_seen - alert.first_seen) / 60) or 1} min · signature: <b>{html.escape(kw)}</b><br>'
        f'<span style="color:#52514e;font-size:.85rem">Tickets: {", ".join(alert.ticket_refs)}{" · " + subj if subj else ""}</span></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- view switch
n_pending = len(hitl.pending(ss.queue))
QUEUE_VIEW = "👩‍💼 Review queue"  # fixed label: a label that changes resets the widget (and the view)
v1, v2 = st.columns([3, 2], vertical_alignment="center")
view = v1.segmented_control("View", ["🎫 Live triage", QUEUE_VIEW], default="🎫 Live triage", key="view",
                            label_visibility="collapsed")
n_oncall = sum(i["status"] == hitl.ONCALL for i in hitl.pending(ss.queue))
v2.markdown(f"**{n_pending}** waiting for review" + (f" · 🚨 **{n_oncall}** on-call" if n_oncall else ""))
if view == QUEUE_VIEW:
    if st.button(f"Stream {len(STREAM_TICKETS)} demo tickets into the queue"):
        with st.spinner("Processing ..."):
            for s_, b_ in STREAM_TICKETS:
                record(*desk.process(s_, b_, cfg, k=top_k, cross_lingual=cross_lingual))
        st.rerun()
    labels = desk.triage.labels if desk.triage else {}
    hitl.render(st, ss.queue, labels, auto_count=sum(h["status"] == hitl.AUTO for h in ss.history))
    st.stop()


# ----------------------------------------------------------------------------- input
def _load_example():
    slug = LABELS.get(ss.example)
    if slug:
        _, ss.subject, ss.body = EXAMPLES[slug]


def _run(subject: str, body: str) -> None:
    """Runs the real pipeline and lights up the strip as each stage finishes (pause = presentation only)."""
    done: dict = {}

    def on_stage(key, a, gate=None, d=None):
        done[key] = (*flow.summary(key, a, gate, d), a["timings_ms"].get(key))
        nxt = next((k for k, _, _ in flow.STAGES if k not in done), None)
        flow_slot.markdown(flow.render(done, running=nxt), unsafe_allow_html=True)
        if stage_pause and nxt:
            time.sleep(stage_pause)

    flow_slot.markdown(flow.render({}, running="mask"), unsafe_allow_html=True)
    a = desk.analyse(subject, body, k=top_k, cross_lingual=cross_lingual, on_stage=on_stage)
    ss.analysis = a
    record(a, *desk.decide(a, cfg, on_stage=on_stage))


qp = st.query_params.get("example")
if qp in EXAMPLES and ss.get("_qp") != qp:  # ?example=<slug> preloads and runs a demo ticket
    ss._qp = qp
    _, ss.subject, ss.body = EXAMPLES[qp]
    ss.example = EXAMPLES[qp][0]
    ss._pending_run = True  # run below the form, where the pipeline strip lives

st.selectbox("Load an example", [PLACEHOLDER] + list(LABELS), key="example", on_change=_load_example,
             label_visibility="collapsed")
with st.form("ticket"):
    subject = st.text_input("Subject (optional)", key="subject", placeholder="e.g. VPN gateway down")
    body = st.text_area("Ticket body", key="body", height=140, placeholder="Paste the raw customer message (English or German) ...")
    submitted = st.form_submit_button("Triage ticket", type="primary", width="stretch")
flow_slot = st.empty()  # live pipeline strip

if ss._pending_run:
    ss._pending_run = False
    _run(ss.subject, ss.body)
if submitted:
    if not (subject.strip() or body.strip()):
        st.warning("Enter a subject or a body.")
    else:
        _run(subject, body)
        if ss.analysis["burst"]:
            st.rerun()  # repaint the banner at the top

a = ss.analysis


def bars(probs: dict[str, float], top_n: int = 10) -> str:
    rows = []
    for i, (label, p) in enumerate(sorted(probs.items(), key=lambda kv: -kv[1])[:top_n]):
        color = "#2a78d6" if i == 0 else "#86b6ef"
        rows.append(f'<div class="sd-bar-row"><span>{html.escape(label)}</span><div class="sd-bar-track">'
                    f'<div class="sd-bar-fill" style="width:{max(p * 100, 1):.1f}%;background:{color}"></div></div>'
                    f'<span class="sd-num">{p:.0%}</span></div>')
    return "".join(rows)


def cite_html(text: str) -> str:
    safe = html.escape(text.strip()).replace("$", "&#36;")
    safe = re.sub(r"\[?(TKT-\d{5})\]?", r'<span class="sd-cite">\1</span>', safe)
    return re.sub(r"\n{2,}", "<br><br>", safe).replace("\n", "<br>")


# ----------------------------------------------------------------------------- results
if a is not None:
    t_dec = time.perf_counter()
    gate, d = desk.decide(a, cfg)
    decide_ms = (time.perf_counter() - t_dec) * 1000
    flow_slot.markdown(flow.render(flow.full(a, gate, d)), unsafe_allow_html=True)
    tri = a["triage"]
    icon, label, color, tint = STATUS_STYLE[gate.status]
    left, right = st.columns([3, 2], gap="large")
    with left:
        reasons = "".join(
            f'<div class="sd-reason"><span class="sd-code">{r.code}</span>{html.escape(r.message)}</div>' for r in gate.reasons
        ) or '<div class="sd-reason">All checks passed: confident triage, strong precedent, no risk signals.</div>'
        st.markdown(f'<div class="sd-badge" style="border-color:{color};background:{tint}">'
                    f'<h3 style="color:{color}">{icon} {label}</h3>{reasons}</div>', unsafe_allow_html=True)

        mode_label = {"llm": "LLM draft (verified)", "extractive": "Offline extractive draft",
                      "holding": "Holding reply (no grounded content)"}[d.mode]
        st.markdown(f"**Draft reply** · {mode_label} · language `{a['language']}` · cites "
                    + (", ".join(f"`{c}`" for c in d.cited_ticket_ids) or "none"))
        st.markdown(f'<div class="sd-draft">{cite_html(d.reply)}</div>', unsafe_allow_html=True)
        if d.issues:
            st.error("Grounding guard: " + " · ".join(d.issues))
        if d.unsupported_requests:
            st.warning("Not grounded in precedent - left for the agent: " + " · ".join(d.unsupported_requests))
        if d.internal_note:
            st.caption(f"Internal note: {d.internal_note}")
        for n in gate.notes:
            st.caption(f"ℹ️ {n}")
        with st.expander("Copy / edit draft"):
            st.text_area("Draft", d.reply, height=220, label_visibility="collapsed", key=f"draft_{a['ref']}")

    with right:
        c1, c2 = st.columns(2)
        for col, head, title in ((c1, "type", "Type"), (c2, "priority", "Priority")):
            col.markdown(f'<div class="sd-card"><div class="sd-k">{title}</div><div class="sd-v">{tri[head]["label"]}</div>'
                         f'<div class="sd-conf">confidence {tri[head]["confidence"]:.0%}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sd-card"><div class="sd-k">Queue</div><div class="sd-v">{tri["queue"]["label"]}</div>'
                    f'<div class="sd-conf">confidence {tri["queue"]["confidence"]:.0%}</div></div>', unsafe_allow_html=True)
        c3, c4, c5 = st.columns(3)
        c3.markdown(f'<div class="sd-card"><div class="sd-k">Language</div><div class="sd-v">{a["language"].upper()}</div>'
                    f'<div class="sd-conf">auto-detected</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="sd-card"><div class="sd-k">Critical</div><div class="sd-v">{tri["critical"]["score"]:.0%}</div>'
                    f'<div class="sd-conf">threshold {cfg.critical_threshold:.0%}</div></div>', unsafe_allow_html=True)
        total = a["timings_ms"]["total"] + sum(a["timings_ms"].get(k, 0) for k in ("gate", "draft", "verify"))
        c5.markdown(f'<div class="sd-card"><div class="sd-k">Latency</div><div class="sd-v">{total:.0f} ms</div>'
                    f'<div class="sd-conf">end-to-end</div></div>', unsafe_allow_html=True)
        stages = {k: v for k, v in a["timings_ms"].items() if k != "total"}
        st.caption("Stage latency: " + " · ".join(f"{k} {v:.0f} ms" if v >= 1 else f"{k} <1 ms" for k, v in stages.items()))

    tab_prec, tab_triage, tab_guard, tab_queue = st.tabs(
        ["📚 Precedents", "🧮 Triage detail", "🛡️ Guardrails", "📜 Session log"])

    with tab_prec:
        st.caption(f"Top-{len(a['precedents'])} resolved KB tickets (search: {desk.sources.get('Search')}, cosine shown). "
                   f"Precedent floor σ: {cfg.min_similarity:.3f}.")
        for p in a["precedents"]:
            cited = p["ticket_id"] in d.cited_ticket_ids
            with st.container(border=True):
                st.markdown(
                    f'<div class="sd-prec-head"><span><b>{p["ticket_id"]}</b> {"· <span class=sd-cite>cited</span>" if cited else ""} '
                    f'<span class="sd-chip">{p["language"].upper()}</span><span class="sd-chip">{p["type"]}</span>'
                    f'<span class="sd-chip">{p["queue"]}</span><span class="sd-chip">{p["priority"]}</span></span>'
                    f'<span class="sd-sim">similarity <b>{p["similarity"]:.3f}</b></span></div>',
                    unsafe_allow_html=True)
                st.progress(min(max((p["similarity"] - 0.75) / 0.25, 0.0), 1.0))
                if p.get("subject"):
                    st.markdown(f"**{html.escape(p['subject'])}**")
                clean = clean_resolution(p.get("answer"))
                snippet = clean[:320] + ("…" if len(clean) > 320 else "")
                st.markdown(f"*Resolution:* {html.escape(snippet).replace('$', '&#36;')}")
                with st.expander("Full ticket and resolution"):
                    if p.get("body"):
                        st.markdown("**Customer message**")
                        st.text(p["body"])
                    st.markdown("**Agent resolution**")
                    st.text((p.get("answer") or "").replace("\\n", "\n"))
                    if p.get("tags"):
                        st.caption("Tags: " + ", ".join(p["tags"]))

    with tab_triage:
        cols = st.columns(3)
        for col, head in zip(cols, ("type", "queue", "priority")):
            col.markdown(f"**{head.title()}**")
            col.markdown(bars(tri[head]["probs"]), unsafe_allow_html=True)
        expl = tri.get("neighbours", {}).get("explanation") or {}
        if expl:
            st.markdown("**Similar training tickets:** " + " · ".join(html.escape(str(v)) for v in expl.values()))
        st.caption({"full": "Calibrated ensemble: e5 embeddings + LogReg, TF-IDF + LogReg and neighbour votes "
                            "(sentineldesk/triage). Bars: darker = predicted class.",
                    "fallback": "Fallback mode: e5 embeddings + LogReg with temperature scaling.",
                    "mock": "Mock triage: train sentineldesk/triage to see real output."}.get(tri.get("mode"), "")
                   + " Critical score = triage's dedicated critical detector (P of a high-priority Incident).")

    with tab_guard:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**PII masking** (what triage, search and any LLM see)")
            st.code(a["masked_text"][:1500].strip(), language=None, wrap_lines=True)
            st.caption("Masked: " + (", ".join(f"{k} ×{v}" for k, v in a["pii_counts"].items()) or "nothing detected"))
        with g2:
            st.markdown("**Risk signals**")
            st.json({
                "outage_keywords (critical detector input)": tri["critical"].get("keyword_hits") or [],
                "sensitive_requests": a["sensitive_hits"] or {},
                "prompt_injection": a["injection_hits"] or [],
                "language": {"code": a["language"], "confidence": a["language_confidence"]},
            })

    with tab_queue:
        st.caption(f"Every ticket this session. Tickets needing a human are in **{QUEUE_VIEW}** (switch at the top).")
        if ss.history:
            st.dataframe(pd.DataFrame(ss.history), width="stretch", hide_index=True)

    st.divider()
    trace.render(st, a, gate, d, cfg, bars, next((i for i in ss.queue if i["ref"] == a["ref"]), None))
else:
    st.info("Paste a ticket or load an example to see triage, escalation, precedents and the grounded draft.")
    if ss.history:
        st.dataframe(pd.DataFrame(ss.history), width="stretch", hide_index=True)
