"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Helper functions for sidebar history ──────────────────────────────────
def get_recent_threads(limit: int = 20) -> list[dict]:
    """Retrieve recent review sessions from audit_events."""
    try:
        conn = sqlite3.connect(db_path())
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   MAX(risk_level)       AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_thread_events(thread_id: str) -> list[dict]:
    """Retrieve all audit events for a thread."""
    try:
        conn = sqlite3.connect(db_path())
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT timestamp, action, confidence, risk_level, reviewer_id, decision, reason, execution_time_ms
              FROM audit_events
             WHERE thread_id = ?
             ORDER BY id
            """,
            (thread_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review Agent", layout="wide", page_icon="🤖")
st.title("🤖 HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent Sessions")
    threads = get_recent_threads()
    if not threads:
        st.caption("No previous sessions recorded yet.")
    else:
        for t in threads:
            tid = t["thread_id"]
            tid_short = tid[:8]
            risk = t["worst_risk"]
            pr_short = t["pr_url"].split("github.com/")[-1] if "github.com/" in t["pr_url"] else t["pr_url"]
            
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**{pr_short}**")
                st.caption(f"ID: `{tid_short}` | Risk: `{risk}` | Events: {t['events']}")
            with col_b:
                if st.button("View", key=f"view_{tid}"):
                    st.session_state.thread_id = tid
                    st.session_state.pr_url = t["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()
            st.divider()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "GitHub PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"⚠️ Approval Requested — Confidence {conf:.0%}")
    st.info(f"**Reasoning:** {payload['confidence_reasoning']}")
    st.markdown(f"**Summary:** {payload['summary']}")

    comments = payload.get("comments", [])
    if comments:
        st.markdown("### Proposed Review Comments:")
        for c in comments:
            st.markdown(f"- **[{c['severity'].upper()}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Show Diff Preview", expanded=False):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Reviewer Feedback (optional note to include with the comment):", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    
    if col1.button("✅ Approve & Post", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit / Suggest Changes", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"🚨 Strong Escalation — Confidence {conf:.0%}")
    st.warning(f"**Reasoning:** {payload['confidence_reasoning']}")
    
    if payload.get("risk_factors"):
        st.error("**Identified Risk Factors:**\n" + "\n".join(f"- {rf}" for rf in payload["risk_factors"]))
    
    st.markdown(f"**Summary:** {payload['summary']}")

    with st.form("escalation_form"):
        st.markdown("### The agent requires clarification before proceeding:")
        answers: dict[str, str] = {}
        questions = payload.get("questions", [])
        for i, q in enumerate(questions):
            answers[q] = st.text_area(f"Q{i+1}: {q}", key=f"q_{i}", height=70)
        
        submitted_answers = st.form_submit_button("Submit Clarifications & Re-synthesize", type="primary")
        if submitted_answers:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg
            )
        else:
            result = await app.ainvoke(
                Command(resume=resume_value), cfg
            )
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + analyzing with LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming agent with your input..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ **{action}** — Comment successfully posted to [{st.session_state.pr_url}]({st.session_state.pr_url})")
    elif action == "rejected":
        st.warning("❌ **Rejected** — No comment posted to GitHub.")
    else:
        st.info(f"Final Action: **{action}**")
    
    st.caption(f"Session thread_id: `{st.session_state.thread_id}` · "
               f"Replay via CLI: `uv run python -m audit.replay --thread {st.session_state.thread_id}`")

# If an existing thread is selected from sidebar, display its timeline
if st.session_state.thread_id and st.session_state.interrupt_payload is None and st.session_state.final is None:
    events = get_thread_events(st.session_state.thread_id)
    if events:
        st.subheader(f"Audit Trail for Thread `{st.session_state.thread_id[:8]}...`")
        st.dataframe(events, use_container_width=True)
