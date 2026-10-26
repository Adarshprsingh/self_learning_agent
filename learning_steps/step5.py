import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"

# The history list: this is the chatbot's "memory"
messages = [
    {"role": "system", "content": "You are a helpful study assistant. Keep answers short and clear."}
]

print("Chatbot ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    # 1. Add your message to the history
    messages.append({"role": "user", "content": user_input})

    # 2. Send the WHOLE history to the AI
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    reply = response.choices[0].message.content

    # 3. Add the AI's reply to the history too
    messages.append({"role": "assistant", "content": reply})

    print("AI:", reply, "\n")