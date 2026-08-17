"""IterRet project constants and configuration defaults.

This module centrally manages all magic numbers, default values, and literals
used throughout the IterRet codebase. Single source of truth for tuning parameters.
"""

# LLM Configuration Defaults
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_LLM_MAX_TOKENS = 1024

# Experiment Execution Limits
DEFAULT_MAX_ITERATIONS = 5  # Maximum reasoning loop iterations
DEFAULT_MAX_STUCK_REFLECTS = 2  # Maximum consecutive reflection steps without progress

# Memory Builder Configuration
DEFAULT_MAX_CHARS_PER_CALL = 4000  # Conservative for 8k-token context servers

# Graph Traversal Limits (active set sizes)
MAX_ACTIVE_CUES = 40  # Maximum cues to track per iteration
MAX_ACTIVE_TAGS = 15  # Maximum tags to activate per iteration
MAX_NEW_CONTENT_PER_ROUND = 25  # Maximum new content nodes per reflection round
MAX_INNER_CTC_ITERATIONS = 3  # Maximum sub-iterations for CTC graph traversal

# Embedding Model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LoCoMo Dataset Categories
CATEGORY_NAMES = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}

# Default evaluation categories (exclude adversarial by default)
DEFAULT_EVAL_CATEGORIES = (1, 2, 3, 4)

# Candidate paths checked (in order) when no --locomo-path / locomo_path override is given.
KNOWN_LOCOMO_PATHS = (
    "MRAgent/data/dataset_locomo.json",
    "Reflective-Experience-for-Memory-Search/data/locomo/locomo10.json",
)
