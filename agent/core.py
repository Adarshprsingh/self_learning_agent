import json
import logging

from openai import OpenAIError

from . import config
from .llm import get_client
from .memory import get_store
from .tools import TOOLS, run_tool

log = logging.getLogger("agent")


def new_conversation():
    return [{"role": "system", "content": ""}]


def build_system_prompt(query):
    store = get_store()
    facts = "\n".join(f"- {f}" for f in store.relevant_facts(query)) or "- (nothing yet)"
    lessons = "\n".join(
        f"- Avoid: {m['mistake']} -> Instead: {m['better_approach']}"
        for m in store.relevant_lessons(query)
    ) or "- (none yet)"

    return (
        f"Your name is {config.AGENT_NAME}. You are a helpful study assistant with tools "
        f"for notes, math and memory. Keep answers short unless told otherwise.\n\n"
        f"What you remember about the user:\n{facts}\n\n"
        f"Relevant lessons from past mistakes (follow them!):\n{lessons}\n\n"
        f"When the user tells you something lasting about themselves "
        f"(name, goals, exams, preferences), use the remember_fact tool."
    )


def run_agent(messages, user_query):
    """Run the agent loop. Returns (answer, tools_used)."""
    checkpoint = len(messages) - 1  # index of the user's message
    messages[0]["content"] = build_system_prompt(user_query)
    tools_used = []

    try:
        client = get_client()
        for _ in range(config.MAX_STEPS):
            response = client.chat.completions.create(
                model=config.MODEL, messages=messages, tools=TOOLS
            )
            message = response.choices[0].message

            # No tool requested -> this is the final answer
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
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = run_tool(name, args)
                except json.JSONDecodeError:
                    args, result = {}, "Error: tool arguments were not valid JSON."

                log.info("Tool %s(%s) -> %s", name, args, result)
                tools_used.append(f"{name}({args})")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return "Stopped: too many steps.", tools_used

    except (OpenAIError, RuntimeError) as e:
        log.exception("Agent run failed")
        del messages[checkpoint:]  # drop the half-finished turn so the chat stays valid
        return f"Sorry, I couldn't get an answer ({type(e).__name__}). Please try again.", tools_used