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


# 1. A normal Python function (the actual "tool")
def save_note(text: str) -> str:
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")
    return "Note saved."


# 2. A description of the tool, so the AI knows it exists
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save an important note for the user to read later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note to save.",
                    }
                },
                "required": ["text"],
            },
        },
    }
]

# 3. Ask something that should trigger the tool
user_message = "Please note that my exam is on Friday."

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": user_message}],
    tools=tools,
)

message = response.choices[0].message

# 4. Did the AI ask to use a tool?
if message.tool_calls:
    call = message.tool_calls[0]
    print("AI wants to call:", call.function.name)
    print("With arguments:", call.function.arguments)

    args = json.loads(call.function.arguments)  # text -> Python dictionary

    if call.function.name == "save_note":
        result = save_note(**args)
        print("Result:", result)
else:
    print("AI answered directly:", message.content)