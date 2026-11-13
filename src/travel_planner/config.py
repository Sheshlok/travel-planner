"""Settings and API keys.

Replaces the Colab `google.colab.userdata` / `getpass` cells: keys are read
from a `.env` file in the project root (see `.env.example`) or from the shell
environment.
"""

import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY")

# Model settings (the notebook used gpt-5.2 at temperature 0.2 everywhere).
# Set TRAVEL_PLANNER_TEMPERATURE to an empty value to leave temperature out of
# API calls; needed for models that only accept the default (e.g. gpt-5).
MODEL_NAME = os.getenv("TRAVEL_PLANNER_MODEL", "gpt-5.2")
_temperature = os.getenv("TRAVEL_PLANNER_TEMPERATURE", "0.2").strip()
TEMPERATURE: float | None = float(_temperature) if _temperature else None

# Seconds to wait on one model request before retrying (no timeout by default in
# the OpenAI client, so a stalled connection could otherwise hang forever)
LLM_TIMEOUT = float(os.getenv("TRAVEL_PLANNER_LLM_TIMEOUT", "180"))
LLM_MAX_RETRIES = int(os.getenv("TRAVEL_PLANNER_LLM_MAX_RETRIES", "2"))


def missing_keys() -> list[str]:
    """Return the names of required API keys that are not set."""
    return [k for k in REQUIRED_KEYS if not os.getenv(k)]


def check_keys() -> None:
    """Raise a clear error if any required API key is missing."""
    missing = missing_keys()
    if missing:
        raise RuntimeError(
            "Missing API keys: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
