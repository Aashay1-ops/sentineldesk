"""Live pipeline strip for the console: one box per stage, lit up as the real pipeline finishes it.

Every summary is built from the real intermediate output (masked counts, language, triage labels, best
precedent, gate reasons, draft mode, verify result). The only cosmetic part is the pause between stages
(app sidebar, "Stage animation"), so the audience can follow the flow; the work itself is not slowed.
"""
from __future__ import annotations

import html

# (key, label, icon) in pipeline order; keys match Desk.analyse / Desk.decide on_stage events
STAGES = [
    ("mask", "PII mask", "🛡️"), ("language", "Language", "🌐"), ("guardrails", "Guardrails", "🚧"),
    ("triage", "Triage", "🧮"), ("retrieve", "Retrieval", "📚"), ("gate", "Gate", "🚦"),
    ("draft", "Draft", "✍️"), ("verify", "Verify", "✅"), ("decision", "Decision", "🏁"),
]
_TONE = {"ok": "#0ca30c", "warn": "#b97f00", "bad": "#d03b3b", "info": "#2a78d6"}

CSS = """
<style>
.sd-flow { display:grid; grid-template-columns: repeat(9, minmax(0,1fr)); gap:.4rem; margin:.4rem 0 1rem 0; }
.sd-st { border:1px solid var(--line); border-radius:10px; padding:.5rem .55rem; background:var(--card); min-height:92px;
         font-size:.74rem; color:var(--ink-2); position:relative; transition: all .25s ease; }
.sd-st .h { font-size:.78rem; font-weight:650; color:var(--ink); margin-bottom:.2rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sd-st .s { line-height:1.3; word-break:break-word; }
.sd-st .ms { position:absolute; right:.45rem; bottom:.3rem; font-variant-numeric:tabular-nums; color:var(--ink-3); font-size:.68rem; }
.sd-st.wait { opacity:.38; }
.sd-st.run { border-color:#2a78d6; box-shadow:0 0 0 3px rgba(42,120,214,.18); animation: sdpulse 0.9s ease-in-out infinite; }
@keyframes sdpulse { 0%,100% { box-shadow:0 0 0 2px rgba(42,120,214,.12);} 50% { box-shadow:0 0 0 6px rgba(42,120,214,.28);} }
@media (max-width: 900px) { .sd-flow { grid-template-columns: repeat(3, minmax(0,1fr)); } }
</style>
"""


def _top(probs: dict) -> str:
    return max(probs.items(), key=lambda kv: kv[1])[0] if probs else "?"


def summary(key: str, a: dict, gate=None, d=None) -> tuple[str, str]:
    """(text, tone) for one stage, from real pipeline output."""
    if key == "mask":
        pc = a.get("pii_counts") or {}
        return (("masked " + ", ".join(f"{k}×{v}" for k, v in pc.items())) if pc else "no PII found",
                "warn" if pc else "ok")
    if key == "language":
        return f"{a['language'].upper()} · conf {a['language_confidence']:.2f}", "info"
    if key == "guardrails":
        hits = [f"injection ({len(a['injection_hits'])})"] if a.get("injection_hits") else []
        hits += [c.replace("_", "/") for c in (a.get("sensitive_hits") or {})]
        return ("⚠ " + ", ".join(hits)) if hits else "no risk signals", "bad" if a.get("injection_hits") else (
            "warn" if hits else "ok")
    if key == "triage":
        t = a["triage"]
        crit = t["critical"]
        return (f"{t['type']['label']} · {t['queue']['label']} · {t['priority']['label']} "
                f"({t['type']['confidence']:.0%}/{t['queue']['confidence']:.0%}/{t['priority']['confidence']:.0%})"
                f" · critical {crit['score']:.0%}", "bad" if crit.get("fires") else "info")
    if key == "retrieve":
        p = a.get("precedents") or []
        if not p:
            return "no precedents", "warn"
        best = max(p, key=lambda x: x["similarity"])
        return f"{len(p)} hits · best {best['ticket_id']} cos {best['similarity']:.3f}", "info"
    if key == "gate":
        codes = [r.code for r in gate.reasons if r.code not in ("DRAFT_GUARD", "LLM_FLAGGED")]
        pre = a.get("gate_pre", gate.status.value)
        return (pre.replace("_", " ").title() + (": " + ", ".join(codes) if codes else ": all clear"),
                {"AUTO_DRAFT_READY": "ok", "HUMAN_REVIEW": "warn"}.get(pre, "bad"))
    if key == "draft":
        mode = {"llm": "LLM draft", "extractive": "offline extractive", "holding": "holding reply"}.get(d.mode, d.mode)
        model = d.timings.get("model") if getattr(d, "timings", None) else None
        cites = ", ".join(d.cited_ticket_ids) or "no citations"
        return f"{mode}{' · ' + model if model else ''} · {cites}", "ok" if d.mode == "llm" else "info"
    if key == "verify":
        if d.issues or "failed verify" in (d.internal_note or ""):
            return f"✗ {len(d.issues) or 'asks'} failed → draft hidden", "bad"
        if d.mode == "llm":
            return "all checks passed (verify.py)", "ok"
        return "rule checks passed (offline draft)", "ok"
    if key == "decision":
        label = {"AUTO_DRAFT_READY": "✅ auto-draft ready", "HUMAN_REVIEW": "⚠️ to review queue",
                 "ESCALATE_ON_CALL": "🚨 page on-call + review queue"}[gate.status.value]
        return label, {"AUTO_DRAFT_READY": "ok", "HUMAN_REVIEW": "warn"}.get(gate.status.value, "bad")
    raise KeyError(key)


def render(done: dict[str, tuple[str, str, float | None]], running: str | None = None) -> str:
    """done: key -> (text, tone, ms). Stages not done are greyed; `running` pulses."""
    boxes = []
    for key, label, icon in STAGES:
        if key in done:
            text, tone, ms = done[key]
            cls, color = "sd-st", _TONE[tone]
            body = f'<div class="s">{html.escape(text)}</div>'
            ms_txt = f'<span class="ms">{ms:.0f} ms</span>' if ms is not None and ms >= 1 else (
                '<span class="ms">&lt;1 ms</span>' if ms is not None else "")
            style = f' style="border-top:4px solid {color}"'
        else:
            cls, style, ms_txt = ("sd-st run" if key == running else "sd-st wait"), "", ""
            body = '<div class="s">running …</div>' if key == running else '<div class="s">—</div>'
        boxes.append(f'<div class="{cls}"{style}><div class="h">{icon} {label}</div>{body}{ms_txt}</div>')
    return '<div class="sd-flow">' + "".join(boxes) + "</div>"


def full(a: dict, gate, d, decide_ms: dict | None = None) -> dict[str, tuple[str, str, float | None]]:
    """All stages from a finished ticket (static re-render after reruns)."""
    ms = {**a.get("timings_ms", {}), **(decide_ms or {})}
    return {k: (*summary(k, a, gate, d), ms.get(k)) for k, _, _ in STAGES}


if __name__ == "__main__":  # self-check: every stage renders from a mock ticket
    from ui import mocks
    from ui.drafting import DraftResult
    from ui.gate import GateConfig, evaluate_gate

    tri = mocks.mock_triage("VPN down", "VPN gateway down since 8am")
    a = {"pii_counts": {"email": 1}, "language": "en", "language_confidence": 0.99, "injection_hits": [],
         "sensitive_hits": {}, "triage": tri, "precedents": [{"ticket_id": "TKT-00001", "similarity": 0.91}],
         "timings_ms": {"mask": 0.2, "triage": 20.0}}
    g = evaluate_gate(tri, a["precedents"], {}, [], GateConfig())
    d = DraftResult(reply="x", mode="extractive", cited_ticket_ids=["TKT-00001"])
    out = render(full(a, g, d))
    assert out.count('class="sd-st"') == len(STAGES) and "TKT-00001" in out and "masked email×1" in out
    assert render({}, running="mask").count("sd-st run") == 1
    print("flow self-check OK")
