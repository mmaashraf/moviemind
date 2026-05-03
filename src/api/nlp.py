import os
import re
import json
from typing import Any, Dict, Tuple
import requests


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
    """Ollama-backed parser with strict schema guardrails and deterministic fallback."""
    fallback = _rule_parse(query)
    fallback["parsed_by"] = "local-llm-fallback"
    fallback["confidence"] = max(0.55, fallback["confidence"] - 0.1)
    fallback["explanation"] = "Local LLM mode selected; using safe fallback parser with strict schema."

    endpoint = os.environ.get("MOVIEMIND_OLLAMA_URL", "http://127.0.0.1:11434")
    model = os.environ.get("MOVIEMIND_OLLAMA_MODEL", "llama3.1:8b")
    timeout_sec = float(os.environ.get("MOVIEMIND_OLLAMA_TIMEOUT_SEC", "12"))

    system_prompt = (
        "You convert movie recommendation requests into JSON.\n"
        "Return ONLY valid JSON with keys: intent, filters, model_hint.\n"
        "intent in [recommend,predict,explain].\n"
        "filters may contain user_id (int), top_n (1..100), genre (string).\n"
        "model_hint may be one of: baseline_global_mean, linear_regression, random_forest, gradient_boosting, ncf_baseline, ncf_tuned.\n"
    )
    user_prompt = f"Query: {query}\nReturn JSON only."
    try:
        resp = requests.post(
            f"{endpoint.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": f"{system_prompt}\n{user_prompt}",
                "stream": False,
                "format": "json",
            },
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        body = resp.json()
        raw_text = str(body.get("response", "")).strip()
        llm_obj = json.loads(raw_text) if raw_text else {}

        # Guardrails: normalize and bound
        intent = str(llm_obj.get("intent", fallback["intent"])).lower()
        if intent not in {"recommend", "predict", "explain"}:
            intent = fallback["intent"]
        filters = llm_obj.get("filters", {}) if isinstance(llm_obj.get("filters", {}), dict) else {}
        if "top_n" in filters:
            filters["top_n"] = _bounded_top_n(filters["top_n"])
        if "user_id" in filters:
            try:
                filters["user_id"] = max(1, int(filters["user_id"]))
            except Exception:
                filters.pop("user_id", None)
        if "genre" in filters:
            g = str(filters["genre"]).strip().lower()
            if g not in SUPPORTED_GENRES:
                filters.pop("genre", None)
            else:
                filters["genre"] = g
        model_hint = llm_obj.get("model_hint")
        if model_hint not in {
            None,
            "baseline_global_mean",
            "linear_regression",
            "random_forest",
            "gradient_boosting",
            "ncf_baseline",
            "ncf_tuned",
        }:
            model_hint = None

        return {
            "parsed_by": "local-llm-ollama",
            "confidence": 0.82 if filters else 0.7,
            "intent": intent,
            "filters": filters,
            "model_hint": model_hint,
            "explanation": f"Parsed via local Ollama model `{model}` with schema guardrails.",
        }
    except Exception as exc:
        fallback["explanation"] += f" Local LLM unavailable/error: {exc}"
        return fallback


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

