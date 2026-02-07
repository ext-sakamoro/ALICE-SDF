"""Configuration for ALICE-SDF Text-to-3D server."""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# LLM Models
CLAUDE_MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
}

GEMINI_MODELS = {
    "flash": "gemini-2.5-flash",
    "pro": "gemini-2.5-pro-preview-06-05",
}

# Default settings
DEFAULT_PROVIDER = "claude"
DEFAULT_MODEL = "haiku"
DEFAULT_RESOLUTION = 64
DEFAULT_BOUNDS = (-5.0, 5.0)

# Cache
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
