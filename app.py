import streamlit as st

from agent import config
from agent.core import new_conversation, run_agent
from agent.memory import get_store
from agent.reflection import reflect

NAME = config.AGENT_NAME

st.set_page_config(page_title=NAME, page_icon="🤖")

if not config.API_KEY:
    st.error("GROQ_API_KEY not found. Copy .env.example to .env and add your key.")
    st.stop()

store = get_store()

# ---------- session state (survives page refreshes) ----------

if "messages" not in st.session_state:
    st.session_state.messages = new_conversation()  # sent to the AI
if "chat" not in st.session_state:
    st.session_state.chat = []  # shown on screen

# ---------- sidebar: what the agent knows ----------

with st.sidebar:
    st.title(f"🤖 {NAME}")

    if st.button("New chat"):
        st.session_state.messages = new_conversation()
        st.session_state.chat = []
        st.rerun()

    facts = store.list_facts()
    st.subheader(f"Facts I remember ({len(facts)})")
    for fid, doc in facts:
        c1, c2 = st.columns([5, 1])
        c1.write(f"• {doc}")
        if c2.button("🗑", key=f"delf_{fid}"):
            store.delete_fact(fid)
            st.rerun()

    lessons = store.list_lessons()
    st.subheader(f"Lessons learned ({len(lessons)})")
    for lid, doc in lessons:
        c1, c2 = st.columns([5, 1])
        c1.write(f"• {doc}")
        if c2.button("🗑", key=f"dell_{lid}"):
            store.delete_lesson(lid)
            st.rerun()

# ---------- main chat ----------

st.title(f"Chat with {NAME}")

for i, item in enumerate(st.session_state.chat):
    with st.chat_message(item["role"]):
        st.write(item["content"])

        if item["role"] != "assistant":
            continue

        if item.get("tools"):
            st.caption("🔧 Tools used: " + ", ".join(item["tools"]))

        status = item.get("feedback")

        if status is None:
            col1, col2, _ = st.columns([1, 1, 8])
            if col1.button("👍", key=f"up_{i}"):
                item["feedback"] = "good"
                st.rerun()
            if col2.button("👎", key=f"down_{i}"):
                item["feedback"] = "asking"
                st.rerun()

        elif status == "asking":
            with st.form(key=f"form_{i}"):
                comment = st.text_input("What was wrong? (more detail = better lesson)")
                submitted = st.form_submit_button(f"Teach {NAME}")
            if submitted:
                with st.spinner("Reflecting on the mistake..."):
                    lesson = reflect(item["question"], item["content"], comment)
                item["feedback"] = "bad"
                item["lesson"] = (
                    f"Avoid: {lesson.mistake} → Instead: {lesson.better_approach}"
                    if lesson
                    else "Could not create a lesson this time."
                )
                st.rerun()

        elif status == "good":
            st.caption("👍 Thanks!")

        elif status == "bad":
            st.caption("📚 Learned: " + item.get("lesson", ""))

prompt = st.chat_input(f"Message {NAME}...")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner(f"{NAME} is thinking..."):
        answer, tools_used = run_agent(st.session_state.messages, prompt)

    st.session_state.chat.append({
        "role": "assistant",
        "content": answer,
        "tools": tools_used,
        "question": prompt,
        "feedback": None,
    })
    st.rerun()
