"""Human-in-the-loop review queue for the console.

Every ticket the gate does not auto-clear (HUMAN_REVIEW or ESCALATE_ON_CALL) lands here with its reasons,
triage, precedents and draft. An agent sends the reply (as drafted or edited), re-routes it (corrected
queue / priority), or takes it over. Every decision is logged; label corrections can be downloaded as
CSV (feedback for the next training run - never mixed into the holdout).

ponytail: the queue lives in the Streamlit session (README: no new services or databases), so a console
restart empties it. Upgrade path: append items/decisions to a JSON-lines file under artifacts/ui/.
"""
from __future__ import annotations

import html
from datetime import datetime

ONCALL, REVIEW, AUTO = "ESCALATE_ON_CALL", "HUMAN_REVIEW", "AUTO_DRAFT_READY"
OUTCOMES = {"sent": "✅ sent as drafted", "edited": "✏️ sent after edit", "rerouted": "↪️ re-routed",
            "taken": "🙋 taken over by agent"}


def enqueue(queue: list[dict], a: dict, gate, d) -> dict | None:
    """Add a ticket the gate did not auto-clear. Idempotent per ref. Returns the item (or None)."""
    if gate.status.value == AUTO or any(it["ref"] == a["ref"] for it in queue):
        return None
    tri = a["triage"]
    item = {
        "ref": a["ref"], "received": datetime.now(), "subject": a["subject"] or "(no subject)", "body": a["body"],
        "language": a["language"], "status": gate.status.value,
        "reasons": [(r.code, r.message) for r in gate.reasons],
        "pred": {h: (tri[h]["label"], tri[h]["confidence"]) for h in ("type", "queue", "priority")},
        "precedents": [(p["ticket_id"], p["similarity"]) for p in a["precedents"]],
        "draft": d.reply, "draft_mode": d.mode, "cites": list(d.cited_ticket_ids),
        "state": "pending", "outcome": None, "final": None, "resolved": None,
    }
    queue.append(item)
    return item


def pending(queue: list[dict]) -> list[dict]:
    """On-call first, then oldest first."""
    return sorted((it for it in queue if it["state"] == "pending"),
                  key=lambda it: (it["status"] != ONCALL, it["received"]))


def resolve(item: dict, action: str, reply: str, queue_label: str, priority: str, note: str = "") -> None:
    if action not in ("send", "reroute", "take"):
        raise ValueError(action)
    if action == "send" and not reply.strip():
        raise ValueError("Reply is empty.")
    outcome = {"send": "edited" if reply.strip() != item["draft"].strip() else "sent",
               "reroute": "rerouted", "take": "taken"}[action]
    item.update(state="resolved", outcome=outcome, resolved=datetime.now(),
                final={"reply": reply if action == "send" else None, "queue": queue_label,
                       "priority": priority, "note": note.strip()})


def log_rows(queue: list[dict]) -> list[dict]:
    rows = []
    for it in sorted((i for i in queue if i["state"] == "resolved"), key=lambda i: i["resolved"], reverse=True):
        f = it["final"]
        rows.append({
            "ref": it["ref"], "gate": it["status"], "outcome": OUTCOMES[it["outcome"]], "subject": it["subject"],
            "queue (pred → final)": f"{it['pred']['queue'][0]} → {f['queue']}",
            "priority (pred → final)": f"{it['pred']['priority'][0]} → {f['priority']}",
            "label corrected": f["queue"] != it["pred"]["queue"][0] or f["priority"] != it["pred"]["priority"][0],
            "handled in (s)": round((it["resolved"] - it["received"]).total_seconds()),
            "note": f["note"],
        })
    return rows


# ---------------------------------------------------------------- Streamlit view
def _age(dt: datetime) -> str:
    s = int((datetime.now() - dt).total_seconds())
    return f"{s}s ago" if s < 60 else f"{s // 60} min ago"


def render(st, queue: list[dict], labels: dict[str, list[str]], auto_count: int) -> None:
    import pandas as pd

    todo, done = pending(queue), [i for i in queue if i["state"] == "resolved"]
    c = st.columns(4)
    c[0].metric("Pending review", len(todo))
    c[1].metric("🚨 On-call pending", sum(i["status"] == ONCALL for i in todo))
    c[2].metric("Resolved by agents", len(done))
    c[3].metric("Auto-drafted (no human)", auto_count)

    if not todo:
        st.info("Queue is empty. Triage a ticket that needs a human, or use **Stream demo tickets** to fill it.")
    for it in todo:
        oncall = it["status"] == ONCALL
        color = "#d03b3b" if oncall else "#b97f00"
        with st.container(border=True):
            st.markdown(
                f'<div style="border-left:5px solid {color};padding-left:.6rem">'
                f'<b>{"🚨 ON-CALL" if oncall else "⚠️ REVIEW"}</b> · <code>{it["ref"]}</code> · '
                f'<b>{html.escape(it["subject"])}</b> · {it["language"].upper()} · '
                f'<span style="color:#7a7974">{_age(it["received"])}</span><br>'
                + "".join(f'<span class="sd-code">{html.escape(code)}</span>{html.escape(msg)}<br>'
                          for code, msg in it["reasons"]) + "</div>", unsafe_allow_html=True)
            left, right = st.columns([3, 2])
            with left:
                with st.expander("Customer message"):
                    st.text(it["body"])
                reply = st.text_area(f"Reply ({it['draft_mode']} draft · cites {', '.join(it['cites']) or 'none'})",
                                     it["draft"], height=170, key=f"hitl_reply_{it['ref']}")
            with right:
                p = it["pred"]
                st.caption(f"Predicted: {p['type'][0]} ({p['type'][1]:.0%}) · {p['queue'][0]} ({p['queue'][1]:.0%}) · "
                           f"{p['priority'][0]} ({p['priority'][1]:.0%})")
                ql, pl = labels.get("queue", [p["queue"][0]]), labels.get("priority", [p["priority"][0]])
                q = st.selectbox("Queue", ql, index=ql.index(p["queue"][0]) if p["queue"][0] in ql else 0,
                                 key=f"hitl_q_{it['ref']}")
                pr = st.selectbox("Priority", pl, index=pl.index(p["priority"][0]) if p["priority"][0] in pl else 0,
                                  key=f"hitl_p_{it['ref']}")
                note = st.text_input("Agent note (optional)", key=f"hitl_n_{it['ref']}")
            b = st.columns(3)
            action = ("send" if b[0].button("✅ Send reply", key=f"hitl_s_{it['ref']}", type="primary", width="stretch")
                      else "reroute" if b[1].button("↪️ Re-route", key=f"hitl_r_{it['ref']}", width="stretch")
                      else "take" if b[2].button("🙋 Take over", key=f"hitl_t_{it['ref']}", width="stretch") else None)
            if action:
                try:
                    resolve(it, action, reply, q, pr, note)
                except ValueError as e:
                    st.error(str(e))
                else:
                    st.toast(f"{it['ref']}: {OUTCOMES[it['outcome']]}")
                    st.rerun()

    if done:
        st.markdown("#### Decision log")
        df = pd.DataFrame(log_rows(queue))
        st.dataframe(df, width="stretch", hide_index=True)
        corrected = df[df["label corrected"]]
        st.caption(f"{len(corrected)} label correction(s) by agents: feedback for the next training run "
                   "(kept apart from the holdout).")
        st.download_button("Download decision log (CSV)", df.to_csv(index=False), "hitl_decisions.csv", "text/csv")


if __name__ == "__main__":  # self-check: queue order, idempotent enqueue, outcomes, log
    from types import SimpleNamespace as NS

    def ticket(ref, status):
        gate = NS(status=NS(value=status), reasons=[NS(code="X", message="m")])
        tri = {h: {"label": "L", "confidence": 0.4} for h in ("type", "queue", "priority")}
        a = {"ref": ref, "subject": "s", "body": "b", "language": "en", "triage": tri, "precedents": []}
        return a, gate, NS(reply="draft", mode="holding", cited_ticket_ids=[])

    q = []
    assert enqueue(q, *ticket("A", AUTO)) is None
    enqueue(q, *ticket("R", REVIEW))
    enqueue(q, *ticket("O", ONCALL))
    enqueue(q, *ticket("O", ONCALL))
    assert [i["ref"] for i in pending(q)] == ["O", "R"]
    resolve(q[0], "send", "draft", "L", "L")
    resolve(q[1], "reroute", "", "Billing", "high")
    assert [i["outcome"] for i in q] == ["sent", "rerouted"] and not pending(q)
    rows = log_rows(q)
    assert sum(r["label corrected"] for r in rows) == 1
    try:
        resolve(q[0], "send", "  ", "L", "L")
        raise AssertionError("empty reply accepted")
    except ValueError:
        pass
    print("hitl self-check OK")
