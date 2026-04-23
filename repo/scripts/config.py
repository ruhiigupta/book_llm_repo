import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            " GROQ_API_KEY not found.\n"
            "   Set it in your .env file:\n"
            "   GROQ_API_KEY=gsk_your_key_here"
        )

    return Groq(api_key=api_key)


def get_base_dir():
    return BASE_DIR