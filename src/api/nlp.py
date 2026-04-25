import os
import re
from typing import Any, Dict, Tuple


SUPPORTED_GENRES = [
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film-noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci-fi",
    "thriller",
    "war",
    "western",
]


def _bounded_top_n(value: Any, default: int = 10) -> int:
    try:
        n = int(value)
    except Exception:
        return default
    return max(1, min(100, n))


def _rule_parse(query: str) -> Dict[str, Any]:
    # Deterministic parser keeps behavior reproducible for evaluation and debugging.
    q = query.strip().lower()
    intent = "recommend"
    filters: Dict[str, Any] = {}
    explanation = "Parsed using deterministic rules."

    user_match = re.search(r"(?:user|userid)\s*(\d+)", q)
    if user_match:
        filters["user_id"] = int(user_match.group(1))

    top_match = re.search(r"(?:top|recommend|show)\s*(\d+)", q)
    if top_match:
        filters["top_n"] = _bounded_top_n(top_match.group(1))

    for genre in SUPPORTED_GENRES:
        if genre in q:
            filters["genre"] = genre
            break

    if "explain" in q:
        intent = "explain"
    elif "predict" in q or "rating for" in q:
        intent = "predict"

    model_hint = None
    if "gradient boosting" in q or "gb" in q:
        model_hint = "gradient_boosting"
    elif "random forest" in q or "rf" in q:
        model_hint = "random_forest"
    elif "linear" in q:
        model_hint = "linear_regression"
    elif "tuned" in q:
        model_hint = "ncf_tuned"
    elif "dl" in q or "ncf" in q:
        model_hint = "ncf_baseline"
    elif "baseline" in q:
        model_hint = "baseline_global_mean"

    return {
        "parsed_by": "rule-parser",
        "confidence": 0.9 if filters else 0.65,
        "intent": intent,
        "filters": filters,
        "model_hint": model_hint,
        "explanation": explanation,
    }


def _local_llm_parse(query: str) -> Dict[str, Any]:
    # Guardrailed local mode: currently deterministic extraction plus lower confidence.
    # This keeps behavior safe/reproducible until a concrete local model endpoint is wired.
    parsed = _rule_parse(query)
    parsed["parsed_by"] = "local-llm-fallback"
    parsed["confidence"] = max(0.55, parsed["confidence"] - 0.1)
    parsed["explanation"] = "Local LLM mode selected; using safe fallback parser with strict schema."
    return parsed


def _api_llm_parse(query: str) -> Dict[str, Any]:
    # Optional mode placeholder to keep runtime contract stable.
    parsed = _rule_parse(query)
    parsed["parsed_by"] = "api-llm-fallback"
    parsed["confidence"] = max(0.6, parsed["confidence"] - 0.05)
    parsed["explanation"] = "API LLM mode selected but API client not configured; used guarded fallback parser."
    return parsed


def parse_query(query: str, runtime_mode: str) -> Tuple[str, Dict[str, Any]]:
    runtime_mode = runtime_mode.lower()
    if runtime_mode == "rule-only":
        return runtime_mode, _rule_parse(query)
    if runtime_mode == "local-llm":
        return runtime_mode, _local_llm_parse(query)
    if runtime_mode == "api-llm":
        if not os.environ.get("MOVIEMIND_API_LLM_ENABLED"):
            # Guardrail: if API mode is selected but not configured, return a safe fallback
            # instead of failing the request.
            payload = _api_llm_parse(query)
            payload["explanation"] += " Enable MOVIEMIND_API_LLM_ENABLED=1 to activate API client path."
            return runtime_mode, payload
        return runtime_mode, _api_llm_parse(query)
    return "rule-only", _rule_parse(query)

