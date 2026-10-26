import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")
AGENT_NAME = os.getenv("AGENT_NAME", "Nova")

NOTES_FILE = BASE_DIR / "notes.txt"
DB_PATH = BASE_DIR / "chroma_db"
OLD_MEMORY_FILE = BASE_DIR / "memory.json"
MAX_STEPS = 5

# Logging goes to agent.log so errors can be investigated later
log = logging.getLogger("agent")
if not log.handlers:
    log.setLevel(logging.INFO)
    _handler = logging.FileHandler(BASE_DIR / "agent.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_handler)