from openai import OpenAI

from  . import config

_client = None


def get_client():
    global _client
    if _client is None:
        if not config.API_KEY:
            raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")
        _client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    return _client