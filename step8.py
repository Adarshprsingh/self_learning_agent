import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"
NOTES_FILE = "notes.txt"
MAX_STEPS = 5  # safety limit so the loop can never run forever


# ---------- TOOLS (normal Python functions) ----------

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
    # Only allow numbers and basic math symbols, for safety
    allowed = "0123456789+-*/(). "
    if not all(ch in allowed for ch in expression):
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


# Maps a tool name (text) to the real function
TOOL_FUNCTIONS = {
    "save_note": save_note,
    "search_notes": search_notes,
    "calculate": calculate,
}

# ---------- TOOL DESCRIPTIONS (what the AI sees) ----------

tools = [
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save an important note for the user to read later.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The note to save."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the saved notes for a keyword and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string", "description": "Word to look for."}},
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a math expression such as '23 * 4 + 1'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "The math expression."}},
                "required": ["expression"],
            },
        },
    },
]


# ---------- THE AGENT LOOP ----------

def run_agent(messages):
    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message

        # No tool requested -> this is the final answer
        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content})
            return message.content

        # Tool requested -> record the request in the history
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

        # Run each requested tool and add its result to the history
        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [tool] {name}({args})")

            if name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[name](**args)
            else:
                result = f"Error: unknown tool {name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # loop continues: the AI now sees the tool results

    return "Stopped: too many steps."


# ---------- CHAT ----------

messages = [
    {
        "role": "system",
        "content": "You are a helpful study assistant with tools for notes and math. "
                   "Use tools when they help. Keep answers short.",
    }
]

print("Agent ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})
    answer = run_agent(messages)
    print("AI:", answer, "\n")