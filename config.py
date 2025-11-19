import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key
if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    OPENAI_API_KEY = None

# Vector Store Configuration
VECTOR_STORE_PATH = "vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# App Configuration
APP_TITLE = "🧠 CramAI – Study Smarter, Not Harder"
APP_ICON = "📚"

# Quiz Configuration
QUIZ_DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
QUIZ_TYPES = ["MCQ", "Objective", "Detailed Q&A", "Numerical/Problem-solving"]

# Summary Types
SUMMARY_TYPES = ["Quick Recap", "Deep Dive", "Flashcard Format"]
