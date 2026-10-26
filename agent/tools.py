import ast
import logging
import operator

from . import config
from .memory import get_store

log = logging.getLogger("agent")


# ---------- safe calculator (no eval) ----------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception:
        return "Error: only numbers and + - * / % ( ) are supported."


# ---------- notes and memory ----------

def save_note(text: str) -> str:
    with open(config.NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    return "Note saved."


def search_notes(keyword: str) -> str:
    if not config.NOTES_FILE.exists():
        return "No notes yet."
    with open(config.NOTES_FILE, encoding="utf-8") as f:
        lines = [line.strip() for line in f if keyword.lower() in line.lower()]
    return "\n".join(lines) if lines else "No matching notes."


def remember_fact(fact: str) -> str:
    get_store().add_fact(fact)
    return "Fact remembered."


TOOL_FUNCTIONS = {
    "save_note": save_note,
    "search_notes": search_notes,
    "calculate": calculate,
    "remember_fact": remember_fact,
}


def _make_tool(name, description, param, param_description):
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


TOOLS = [
    _make_tool("save_note", "Save a note or reminder for the user.", "text", "The note."),
    _make_tool("search_notes", "Search saved notes for a keyword.", "keyword", "Word to look for."),
    _make_tool("calculate", "Calculate a math expression like '23 * 4 + 1'.", "expression", "The expression."),
    _make_tool("remember_fact", "Permanently remember a lasting fact about the user.", "fact", "One short fact."),
]


def run_tool(name, args):
    """Run a tool safely. A broken tool must never crash the agent."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Error: unknown tool {name}"
    try:
        return func(**args)
    except TypeError as e:
        return f"Error: bad arguments ({e})"
    except Exception as e:
        log.exception("Tool %s failed", name)
        return f"Error: {e}"