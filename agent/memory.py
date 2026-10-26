import hashlib
import json
import logging

import chromadb
from pydantic import BaseModel

from . import config

log = logging.getLogger("agent")


class Lesson(BaseModel):
    mistake: str
    better_approach: str


def _make_id(text):
    # Same text -> same id, so duplicates overwrite instead of piling up
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class MemoryStore:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(config.DB_PATH))
        self.facts = client.get_or_create_collection("facts")
        self.lessons = client.get_or_create_collection("lessons")
        self._import_old_memory()

    def _import_old_memory(self):
        """One-time import of memory.json from the early steps."""
        if not config.OLD_MEMORY_FILE.exists():
            return
        if self.facts.count() > 0 or self.lessons.count() > 0:
            return
        with open(config.OLD_MEMORY_FILE, encoding="utf-8") as f:
            old = json.load(f)
        for fact in old.get("facts", []):
            self.add_fact(fact)
        for item in old.get("lessons", []):
            self.add_lesson("general", Lesson(**item))
        log.info("Imported old memory.json")

    # ----- writing -----

    def add_fact(self, fact):
        self.facts.upsert(ids=[_make_id(fact)], documents=[fact])
        log.info("Fact saved: %s", fact)

    def add_lesson(self, question, lesson):
        doc = f"Situation: {question}. Avoid: {lesson.mistake}. Instead: {lesson.better_approach}"
        self.lessons.upsert(
            ids=[_make_id(doc)], documents=[doc], metadatas=[lesson.model_dump()]
        )
        log.info("Lesson saved: %s", doc)

    # ----- searching by meaning -----

    @staticmethod
    def _search(collection, query, k):
        count = collection.count()
        if count == 0:
            return {"documents": [], "metadatas": []}
        result = collection.query(query_texts=[query], n_results=min(k, count))
        return {
            "documents": result["documents"][0],
            "metadatas": result["metadatas"][0] if result["metadatas"] else [],
        }

    def relevant_facts(self, query, k=5):
        # Few facts: include all. Many facts: only the most relevant ones.
        if self.facts.count() <= 10:
            return self.facts.get()["documents"]
        return self._search(self.facts, query, k)["documents"]

    def relevant_lessons(self, query, k=3):
        hits = self._search(self.lessons, query, k)["metadatas"]
        return [m for m in hits if m]

    # ----- for the sidebar -----

    def list_facts(self):
        data = self.facts.get()
        return list(zip(data["ids"], data["documents"]))

    def list_lessons(self):
        data = self.lessons.get()
        return list(zip(data["ids"], data["documents"]))

    def delete_fact(self, fact_id):
        self.facts.delete(ids=[fact_id])

    def delete_lesson(self, lesson_id):
        self.lessons.delete(ids=[lesson_id])


_store = None


def get_store():
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store