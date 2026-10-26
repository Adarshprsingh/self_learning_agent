import os
import json
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
MEMORY_FILE = "memory.json"
MAX_STEPS = 5
MAX_LESSONS_IN_PROMPT = 10  # keep the prompt from growing forever


# ---------- STRUCTURED LESSON (from Step 6) ----------

class Lesson(BaseModel):
    mistake: str
    better_approach: str


# ---------- MEMORY ----------

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"facts": [], "lessons": []}


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def build_system_prompt():
    memory = load_memory()
    facts = "\n".join(f"- {fact}" for fact in memory["facts"]) or "- (nothing yet)"

    recent_lessons = memory["lessons"][-MAX_LESSONS_IN_PROMPT:]
    lessons = "\n".join(
        f"- Avoid: {l['mistake']} -> Instead: {l['better_approach']}"
        for l in recent_lessons
    ) or "- (none yet)"

    return (
        f"Your name is {AGENT_NAME}. You are a helpful study assistant with tools "
        f"for notes, math and memory. Keep answers short unless told otherwise.\n\n"
        f"What you remember about the user:\n{facts}\n\n"
        f"Lessons learned from past mistakes (follow them!):\n{lessons}\n\n"
        f"When the user tells you something lasting about themselves "
        f"(name, goals, exams, preferences), use the remember_fact tool."
    )


# ---------- SELF-REFLECTION ----------

def reflect(question, answer, comment):
    """Ask the AI to write a lesson about a bad answer, then save it."""
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

    memory = load_memory()
    memory["lessons"].append(lesson.model_dump())
    save_memory(memory)
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
    memory = load_memory()
    if fact in memory["facts"]:
        return "Already remembered."
    memory["facts"].append(fact)
    save_memory(memory)
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

def run_agent(messages):
    messages[0]["content"] = build_system_prompt()

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

messages = [{"role": "system", "content": build_system_prompt()}]

print(f"{AGENT_NAME} is ready. Commands: quit, /memory, /lessons\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if user_input == "/memory":
        print(json.dumps(load_memory()["facts"], indent=2), "\n")
        continue

    if user_input == "/lessons":
        print(json.dumps(load_memory()["lessons"], indent=2), "\n")
        continue

    messages.append({"role": "user", "content": user_input})
    answer = run_agent(messages)
    print(f"{AGENT_NAME}:", answer, "\n")

    # ----- FEEDBACK LOOP -----
    feedback = input("Helpful? (y / n / Enter to skip): ").strip().lower()
    if feedback == "n":
        comment = input("What was wrong? (optional, Enter to skip): ").strip()
        print("  Reflecting on the mistake...")
        reflect(user_input, answer, comment)
        print()