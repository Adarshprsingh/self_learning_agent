import os
import json
import hashlib
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"
AGENT_NAME = "Nova"
NOTES_FILE = "notes.txt"
OLD_MEMORY_FILE = "memory.json"
MAX_STEPS = 5

# ---------- VECTOR DATABASE ----------

chroma = chromadb.PersistentClient(path="chroma_db")
facts_col = chroma.get_or_create_collection("facts")
lessons_col = chroma.get_or_create_collection("lessons")


class Lesson(BaseModel):
    mistake: str
    better_approach: str


def make_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def add_fact(fact):
    facts_col.upsert(ids=[make_id(fact)], documents=[fact])


def add_lesson(question, lesson):
    document = (
        f"Situation: {question}. "
        f"Avoid: {lesson.mistake}. Instead: {lesson.better_approach}"
    )
    lessons_col.upsert(
        ids=[make_id(document)],
        documents=[document],
        metadatas=[lesson.model_dump()],
    )


def find_relevant(collection, query, k):
    count = collection.count()
    if count == 0:
        return {"documents": [], "metadatas": []}
    result = collection.query(query_texts=[query], n_results=min(k, count))
    return {
        "documents": result["documents"][0],
        "metadatas": result["metadatas"][0] if result["metadatas"] else [],
    }


def import_old_memory():
    if not os.path.exists(OLD_MEMORY_FILE):
        return
    if facts_col.count() > 0 or lessons_col.count() > 0:
        return
    with open(OLD_MEMORY_FILE, encoding="utf-8") as f:
        old = json.load(f)
    for fact in old.get("facts", []):
        add_fact(fact)
    for item in old.get("lessons", []):
        add_lesson("general", Lesson(**item))
    print("(Imported your old memory.json into the vector database.)")


# ---------- PROMPT WITH ONLY RELEVANT MEMORIES ----------

def build_system_prompt(query):
    if facts_col.count() <= 10:
        facts_list = facts_col.get()["documents"]
    else:
        facts_list = find_relevant(facts_col, query, 5)["documents"]
    facts = "\n".join(f"- {f}" for f in facts_list) or "- (nothing yet)"

    lesson_hits = find_relevant(lessons_col, query, 3)["metadatas"]
    lessons = "\n".join(
        f"- Avoid: {m['mistake']} -> Instead: {m['better_approach']}"
        for m in lesson_hits
        if m
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
        print("  (Could not create a lesson this time.)")
        return

    add_lesson(question, lesson)
    print(f"  [learned] Avoid: {lesson.mistake}")
    print(f"            Instead: {lesson.better_approach}")


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
    messages[0]["content"] = build_system_prompt(user_query)

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content})
            return message.content

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
            print(f"  [tool] {name}({args})")

            if name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[name](**args)
            else:
                result = f"Error: unknown tool {name}"

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Stopped: too many steps."


# ---------- CHAT ----------

import_old_memory()
messages = [{"role": "system", "content": ""}]

print(f"{AGENT_NAME} is ready. Commands: quit, /memory, /lessons\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if user_input == "/memory":
        print(json.dumps(facts_col.get()["documents"], indent=2), "\n")
        continue

    if user_input == "/lessons":
        print(json.dumps(lessons_col.get()["documents"], indent=2), "\n")
        continue

    messages.append({"role": "user", "content": user_input})
    answer = run_agent(messages, user_input)
    print(f"{AGENT_NAME}:", answer, "\n")

    feedback = input("Helpful? (y / n / Enter to skip): ").strip().lower()
    if feedback == "n":
        comment = input("What was wrong? (optional, Enter to skip): ").strip()
        print("  Reflecting on the mistake...")
        reflect(user_input, answer, comment)
        print()