import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"


# 1. Define the shape of the data we want back
class Lesson(BaseModel):
    mistake: str
    better_approach: str


# 2. Ask the AI to reply in JSON with exactly those fields
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You reply only in JSON with exactly two keys: "
                "'mistake' and 'better_approach'. Both values are short strings."
            ),
        },
        {
            "role": "user",
            "content": "A student wrongly said Python lists are immutable. Write a lesson about this.",
        },
    ],
    response_format={"type": "json_object"},
)

raw_text = response.choices[0].message.content
print("Raw reply from AI:")
print(raw_text)
print()

# 3. Convert the text into a real Python object, with validation
try:
    lesson = Lesson.model_validate_json(raw_text)
    print("Mistake:", lesson.mistake)
    print("Better approach:", lesson.better_approach)
except ValidationError as e:
    print("The AI broke the format:", e)