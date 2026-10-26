import logging

from openai import OpenAIError
from pydantic import ValidationError

from . import config
from .llm import get_client
from .memory import Lesson, get_store

log = logging.getLogger("agent")

REFLECTION_PROMPT = (
    "You review a failed answer from an AI assistant. "
    "Reply only in JSON with exactly two keys: 'mistake' and 'better_approach'. "
    "Both are short strings. 'better_approach' must be a general instruction "
    "the assistant can follow in future similar situations."
)


def reflect(question, answer, comment):
    """Create a lesson from a bad answer and save it. Returns the Lesson or None."""
    try:
        response = get_client().chat.completions.create(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": REFLECTION_PROMPT},
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
        lesson = Lesson.model_validate_json(response.choices[0].message.content)
    except (OpenAIError, ValidationError):
        log.exception("Reflection failed")
        return None

    get_store().add_lesson(question, lesson)
    return lesson