import os
import json
import hashlib
import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

MODEL = "openai/gpt-oss-20b"
AGENT_NAME = "Nova"
NOTES_FILE = "notes.txt"
OLD_MEMORY_FILE = "memory.json"
MAX_STEPS = 5

st.set_page_config(page_title=AGENT_NAME, page_icon="🤖")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Check your .env file.")
    st.stop()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


class Lesson(BaseModel):
    mistake: str
    better_approach: str


# ---------- DATABASE ----------

def make_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@st.cache_resource
def get_collections():
    """Open the database once, not on every page refresh."""
    chroma = chromadb.PersistentClient(path="chroma_db")
    facts = chroma.get_or_create_collection("facts")
    lessons = chroma.get_or_create_collection("lessons")

    # One-time import of the old memory.json from earlier steps
    if os.path.exists(OLD_MEMORY_FILE) and facts.count() == 0 and lessons.count() == 0:
        with open(OLD_MEMORY_FILE, encoding="utf-8") as f:
            old = json.load(f)
        for fact in old.get("facts", []):
            facts.upsert(ids=[make_id(fact)], documents=[fact])
        for item in old.get("lessons", []):
            doc = f"Situation: general. Avoid: {item['mistake']}. Instead: {item['better_approach']}"
            lessons.upsert(ids=[make_id(doc)], documents=[doc], metadatas=[item])
    return facts, lessons


facts_col, lessons_col = get_collections()


def add_fact(fact):
    facts_col.upsert(ids=[make_id(fact)], documents=[fact])


def add_lesson(question, lesson):
    doc = f"Situation: {question}. Avoid: {lesson.mistake}. Instead: {lesson.better_approach}"
    lessons_col.upsert(ids=[make_id(doc)], documents=[doc], metadatas=[lesson.model_dump()])


def find_relevant(collection, query, k):
    count = collection.count()
    if count == 0:
        return {"documents": [], "metadatas": []}
    result = collection.query(query_texts=[query], n_results=min(k, count))
    return {
        "documents": result["documents"][0],
        "metadatas": result["metadatas"][0] if result["metadatas"] else [],
    }


# ---------- PROMPT ----------

def build_system_prompt(query):
    if facts_col.count() <= 10:
        facts_list = facts_col.get()["documents"]
    else:
        facts_list = find_relevant(facts_col, query, 5)["documents"]
    facts = "\n".join(f"- {f}" for f in facts_list) or "- (nothing yet)"

    hits = find_relevant(lessons_col, query, 3)["metadatas"]
    lessons = "\n".join(
        f"- Avoid: {m['mistake']} -> Instead: {m['better_approach']}" for m in hits if m
    ) or "- (none yet)"

    return (
        f"Your name is {AGENT_NAME}. You are a helpful study assistant with tools "
        f"for notes, math and memory. Keep answers short unless told otherwise.\n\n"
        f"What you remember about the user:\n{facts}\n\n"
        f"Relevant lessons from past mistakes (follow them!):\n{lessons}\n\n"
        f"When the user tells you something lasting about themselves "
        f"(name, goals, exams, preferences), use the remember_fact tool."
    )


# ---------- SELF-REFLECTION ----------

def reflect(question, answer, comment):
    """Create and save a lesson. Returns the Lesson, or None if it failed."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You review a failed answer from an AI assistant. "
                    "Reply only in JSON with exactly two keys: 'mistake' and "
                    "'better_approach'. Both are short strings. 'better_approach' "
                    "must be a general instruction the assistant can follow in "
                    "future similar situations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question: {question}\n"
                    f"Assistant answer: {answer}\n"
                    f"User feedback: {comment or 'The user said the answer was not helpful.'}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    try:
        lesson = Lesson.model_validate_json(response.choices[0].message.content)
    except ValidationError:
        return None
    add_lesson(question, lesson)
    return lesson


# ---------- TOOLS ----------

def save_note(text: str) -> str:
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    return "Note saved."


def search_notes(keyword: str) -> str:
    if not os.path.exists(NOTES_FILE):
        return "No notes yet."
    with open(NOTES_FILE, encoding="utf-8") as f:
        lines = [line.strip() for line in f if keyword.lower() in line.lower()]
    return "\n".join(lines) if lines else "No matching notes."


def calculate(expression: str) -> str:
    allowed = "0123456789+-*/(). "
    if not all(ch in allowed for ch in expression):
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def remember_fact(fact: str) -> str:
    add_fact(fact)
    return "Fact remembered."


TOOL_FUNCTIONS = {
    "save_note": save_note,
    "search_notes": search_notes,
    "calculate": calculate,
    "remember_fact": remember_fact,
}


def make_tool(name, description, param, param_description):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {param: {"type": "string", "description": param_description}},
                "required": [param],
            },
        },
    }


tools = [
    make_tool("save_note", "Save a note or reminder for the user.", "text", "The note."),
    make_tool("search_notes", "Search saved notes for a keyword.", "keyword", "Word to look for."),
    make_tool("calculate", "Calculate a math expression like '23 * 4 + 1'.", "expression", "The expression."),
    make_tool("remember_fact", "Permanently remember a lasting fact about the user.", "fact", "One short fact."),
]


# ---------- AGENT LOOP ----------

def run_agent(messages, user_query):
    """Returns (answer, list_of_tools_used)."""
    messages[0]["content"] = build_system_prompt(user_query)
    tools_used = []

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        message = response.choices[0].message

        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content})
            return message.content, tools_used

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            tools_used.append(f"{name}({args})")

            if name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[name](**args)
            else:
                result = f"Error: unknown tool {name}"

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Stopped: too many steps.", tools_used


# ---------- SESSION STATE (survives page refreshes) ----------

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": ""}]  # sent to the AI
if "chat" not in st.session_state:
    st.session_state.chat = []  # shown on screen


# ---------- SIDEBAR: what the agent knows ----------

with st.sidebar:
    st.title(f"🤖 {AGENT_NAME}")

    if st.button("New chat"):
        st.session_state.messages = [{"role": "system", "content": ""}]
        st.session_state.chat = []
        st.rerun()

    facts_data = facts_col.get()
    st.subheader(f"Facts I remember ({len(facts_data['ids'])})")
    for fid, doc in zip(facts_data["ids"], facts_data["documents"]):
        c1, c2 = st.columns([5, 1])
        c1.write(f"• {doc}")
        if c2.button("🗑", key=f"delf_{fid}"):
            facts_col.delete(ids=[fid])
            st.rerun()

    lessons_data = lessons_col.get()
    st.subheader(f"Lessons learned ({len(lessons_data['ids'])})")
    for lid, doc in zip(lessons_data["ids"], lessons_data["documents"]):
        c1, c2 = st.columns([5, 1])
        c1.write(f"• {doc}")
        if c2.button("🗑", key=f"dell_{lid}"):
            lessons_col.delete(ids=[lid])
            st.rerun()


# ---------- MAIN CHAT ----------

st.title(f"Chat with {AGENT_NAME}")

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
                comment = st.text_input("What was wrong? (optional, but more detail = better lesson)")
                submitted = st.form_submit_button("Teach " + AGENT_NAME)
            if submitted:
                with st.spinner("Reflecting on the mistake..."):
                    lesson = reflect(item["question"], item["content"], comment)
                item["feedback"] = "bad"
                item["lesson"] = (
                    f"Avoid: {lesson.mistake} → Instead: {lesson.better_approach}"
                    if lesson else "Could not create a lesson this time."
                )
                st.rerun()

        elif status == "good":
            st.caption("👍 Thanks!")

        elif status == "bad":
            st.caption("📚 Learned: " + item.get("lesson", ""))


prompt = st.chat_input(f"Message {AGENT_NAME}...")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner(f"{AGENT_NAME} is thinking..."):
        answer, tools_used = run_agent(st.session_state.messages, prompt)

    st.session_state.chat.append({
        "role": "assistant",
        "content": answer,
        "tools": tools_used,
        "question": prompt,
        "feedback": None,
    })
    st.rerun()