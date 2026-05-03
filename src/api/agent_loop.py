"""
Multi-step tool-calling agent using Ollama /api/chat.

Tools map to MovieMind registry operations so the model can chain:
user summary → recommendations → optional genre filtering (post-retrieval).

Requires a model with Ollama tool support (e.g. llama3.1). MOVIEMIND_OLLAMA_URL / MODEL / TIMEOUT
mirror src/api/nlp.py.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import ReadTimeout

from .genre_helpers import (
    fetch_size_for_genre_filter,
    genre_tokens_from_tool_args,
    movie_matches_genre_tokens,
)
from .nlp import LocalLLMUnavailableError

OLLAMA_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_available_models",
            "description": (
                "List recommendation models and whether each artifact is loaded (available). "
                "Call when you need to choose a valid model_id."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_summary",
            "description": (
                "MovieLens 1M training-era summary: rating counts, top genres, age/occupation when known. "
                "Use for taste context or onboarding-style questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "MovieLens user id (>=1)."},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": (
                "Score unseen movies for a user with a registered model. "
                "Use top_n around 20–40 first if you will filter by genre. "
                "genre_filter keeps rows whose genres string contains that MovieLens token (e.g. action, drama)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Registry id e.g. gradient_boosting, ncf_baseline, ncf_tuned.",
                    },
                    "user_id": {"type": "integer"},
                    "top_n": {"type": "integer", "description": "1..100"},
                    "diversity_alpha": {
                        "type": "number",
                        "description": (
                            "Optional 0..1. Penalizes overlap with genres already seen in higher-ranked picks "
                            "when building the top-N list (same as Manual **Diversity** slider). "
                            "0 = pure predicted rating; higher values (e.g. 0.2–0.5) spread genres more. "
                            "Pass when the user asks for variety, non-repetitive genres, or 'something different'."
                        ),
                    },
                    "genre_filter": {
                        "type": "string",
                        "description": (
                            "Optional. Comma or 'or'-separated genres: e.g. `documentary,fiction` or `documentary or fiction`. "
                            "`fiction` is expanded to narrative genres (not a MovieLens tag). "
                            "Prefer `genre_any` for structured lists."
                        ),
                    },
                    "genre_any": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional. OR-match any of these tokens after alias expansion (e.g. [\"documentary\",\"fiction\"]).",
                    },
                },
                "required": ["model_id", "user_id", "top_n"],
            },
        },
    },
]


def _ollama_config() -> Tuple[str, str, float]:
    endpoint = os.environ.get("MOVIEMIND_OLLAMA_URL", "http://127.0.0.1:11434")
    model = os.environ.get("MOVIEMIND_OLLAMA_MODEL", "llama3.1:8b")
    timeout_sec = float(os.environ.get("MOVIEMIND_AGENT_TIMEOUT_SEC", os.environ.get("MOVIEMIND_OLLAMA_TIMEOUT_SEC", "120")))
    return endpoint.rstrip("/"), model, timeout_sec


def _ollama_read_retries() -> int:
    """Extra attempts after a ReadTimeout on POST .../api/chat (same payload each time)."""
    try:
        n = int(os.environ.get("MOVIEMIND_OLLAMA_READ_RETRIES", "2"))
    except ValueError:
        n = 2
    return max(0, min(n, 5))


def _call_ollama_chat(
    endpoint: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout_sec: float,
) -> Dict[str, Any]:
    """
    Single POST to Ollama /api/chat with tools; retries only on urllib3 ReadTimeout.

    Not in the LLM prompt — this is transport-layer resilience when Ollama is slow to respond.
    """
    extra = _ollama_read_retries()
    attempts = 1 + extra
    url = f"{endpoint}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": OLLAMA_TOOLS,
        "stream": False,
    }
    for attempt in range(attempts):
        try:
            resp = requests.post(url, json=payload, timeout=timeout_sec)
            resp.raise_for_status()
            return resp.json()
        except ReadTimeout as exc:
            if attempt < attempts - 1:
                continue
            raise LocalLLMUnavailableError(
                f"Ollama chat failed after {attempts} attempt(s) (read timeout): {exc}"
            ) from exc
        except Exception as exc:
            raise LocalLLMUnavailableError(f"Ollama chat failed: {exc}") from exc


AGENT_SYSTEM_PROMPT = """You are MovieMind's **ReAct-style** recommendation agent: **Reason** briefly, **Act** only through tools, read **Observations** (tool outputs), then repeat or answer.

## Boundaries (critical)
- **Facts** (ratings, user stats, which movies were recommended, model availability) must come **only** from **tool** results. Do not invent titles, scores, or counts.
- **Language understanding** is encouraged: in your **Thought**, map the user’s casual words to **MovieLens genre tokens** (or to known alias words the server understands, e.g. `fiction` → narrative genres). You may use general knowledge of synonyms—e.g. “scary” → horror, “funny” → comedy, “space” → sci-fi—then pass the **canonical** tokens in `genre_filter` / `genre_any`. If a phrase has no good single tag, pick the closest genre(s) and say what you assumed.
- If the user wants data outside these tools (live web, non-MovieLens catalogs), say so in **Thought** and explain the limit.

## ReAct format (every assistant turn)
1. **Thought:** Start the `content` field with a short paragraph (1–4 sentences). Explain what you know, what you still need, and what tool call(s) you will make next—or that you are ready to give the final answer.
2. **Action:** When you need data, emit **tool_calls** (the chat API supports tools). One turn may include multiple tool calls if independent.
3. **Observation:** After each tool result appears in the conversation, continue in a new turn with another **Thought**, then more **Action** if needed.

Never leave `content` empty on a turn where you call tools—always lead with **Thought:** so the user can follow your reasoning.
Do **not** print lines like `{"name":"get_recommendations",...}` as your whole answer—those are **not** executed. Only native **`tool_calls`** run on the server.

## Tool usage hints
- Taste / onboarding-style questions → often start with **get_user_summary**.
- Unsure which checkpoints exist → **list_available_models** (respect `available`).
- Recommendations → **get_recommendations** with valid `model_id` and `user_id`.
- **Diversity:** If the user wants **varied** genres, less overlap with the same genre buckets, or “not all action” — set **`diversity_alpha`** on **`get_recommendations`** (e.g. **0.15–0.35**). Omit or **0** for maximum score-only ranking. Same semantics as the REST **`diversity_alpha`** / Manual UI slider.
- Genre constraints → set **genre_any** or **genre_filter** (see MovieLens rules below). Request a generous **top_n** (e.g. 30–40) when filtering; the server also over-fetches when filters apply.
- User IDs must be within dataset bounds. If a tool reports an invalid `user_id`, explain the valid range and ask for a valid ID before continuing.

## MovieLens / registry reminders
Model IDs: baseline_global_mean, linear_regression, random_forest, gradient_boosting, ncf_baseline, ncf_tuned (some may be unavailable on disk).

Genres are fixed lowercase tokens: action, adventure, animation, children, documentary, drama, sci-fi, thriller, comedy, romance, crime, horror, mystery, western, war, fantasy, musical, film-noir, ...
There is NO tag literally named "fiction". Use **genre_any** / **genre_filter** with `documentary` and `fiction` (the backend expands `fiction` to narrative genres). Always pass genre parameters when the user asks for specific kinds of movies.

**Final answer:** When tools have given enough data, your last turn should be **Thought** (short recap) plus a clear, friendly answer listing concrete movie titles from tool outputs.
"""


def _execute_tool(registry: Any, name: str, arguments: Dict[str, Any]) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Returns (json_string_for_llm, last_recommendations_if_any)."""
    last_recs: Optional[List[Dict[str, Any]]] = None

    def _user_id_error(uid: int) -> Optional[Dict[str, Any]]:
        min_uid = int(getattr(registry, "min_user_id", 1))
        max_uid = int(getattr(registry, "max_user_id", 1))
        if uid < min_uid or uid > max_uid:
            return {
                "error": "invalid_user_id",
                "user_id": uid,
                "valid_user_id_range": {"min": min_uid, "max": max_uid},
                "hint": "Ask the user for a user_id inside this range.",
            }
        return None

    try:
        if name == "list_available_models":
            rows = []
            for m in registry.list_models():
                rows.append(
                    {
                        "model_id": m.model_id,
                        "display_name": m.display_name,
                        "family": m.family,
                        "available": m.available,
                    }
                )
            return json.dumps({"models": rows}, ensure_ascii=False), None

        if name == "get_user_summary":
            uid = int(arguments["user_id"])
            uid_err = _user_id_error(uid)
            if uid_err is not None:
                return json.dumps(uid_err, ensure_ascii=False), None
            summary = registry.user_summary(uid)
            return json.dumps(summary, ensure_ascii=False, default=str), None

        if name == "get_recommendations":
            mid = str(arguments["model_id"]).strip()
            uid = int(arguments["user_id"])
            uid_err = _user_id_error(uid)
            if uid_err is not None:
                return json.dumps(uid_err, ensure_ascii=False), None
            top_n = int(arguments["top_n"])
            top_n = max(1, min(100, top_n))
            alpha = float(arguments.get("diversity_alpha", 0.0))
            alpha = max(0.0, min(1.0, alpha))

            expanded, unknown_raw = genre_tokens_from_tool_args(arguments)
            has_filter = bool(expanded)
            fetch_n = fetch_size_for_genre_filter(top_n, has_filter)

            recs = registry.recommend(mid, uid, fetch_n, alpha)
            meta: Dict[str, Any] = {
                "requested_top_n": top_n,
                "scored_candidates": len(recs),
                "fetch_n": fetch_n,
                "genre_expanded_tokens": expanded,
                "genre_unknown_phrases": unknown_raw,
            }
            if has_filter:
                recs = [r for r in recs if movie_matches_genre_tokens(r.get("genres"), expanded)]
                meta["after_genre_filter"] = len(recs)
                meta["note"] = (
                    "OR-match: movie kept if any expanded genre token appears in its genres pipe-string. "
                    "`fiction` expands to narrative genres (all except documentary)."
                )
            elif unknown_raw:
                meta["warning"] = (
                    f"No usable genre tokens parsed from {unknown_raw!r}; "
                    "returning unfiltered top scores. Use MovieLens tokens or aliases: fiction, documentary, sci-fi, ..."
                )

            recs = recs[:top_n]
            meta["returned_count"] = len(recs)
            last_recs = recs
            payload = {"recommendations": recs, **meta}
            return json.dumps(payload, ensure_ascii=False), last_recs

        return json.dumps({"error": f"unknown tool {name!r}"}), None
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)}), None


def _normalize_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    if not raw:
        return []
    out = []
    for tc in raw:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            if name:
                out.append({"name": str(name), "arguments": args})
    return out


def _append_tool_message(messages: List[Dict[str, Any]], tool_name: str, content: str) -> None:
    # Field names differ slightly across Ollama versions; include both.
    messages.append({"role": "tool", "name": tool_name, "tool_name": tool_name, "content": content})


_PSEUDO_TOOL_JSON_PATTERN = re.compile(
    r'["\']name["\']\s*:\s*["\'](list_available_models|get_user_summary|get_recommendations)["\']',
    re.IGNORECASE,
)

_PSEUDO_TOOL_JSON_NUDGE = (
    "Your last assistant message looked like JSON tool calls in plain `content`, but the server only executes "
    "tools when you emit native **`tool_calls`** in the chat protocol—text JSON is ignored. "
    "Repeat the same intent using real **`tool_calls`** (with Thought in `content` if you like). "
    "Do not reply with only printed JSON tool lines."
)


def _content_looks_like_pseudo_tool_json(content: str) -> bool:
    """True when the model typed tool-ish JSON in `content` but did not emit structured tool_calls."""
    s = (content or "").strip()
    if len(s) < 24:
        return False
    if _PSEUDO_TOOL_JSON_PATTERN.search(s):
        return True
    if ("name" in s or "'name'" in s) and ("parameters" in s or "arguments" in s) and "{" in s:
        return True
    return False


def _assistant_record_from_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        rec["tool_calls"] = msg["tool_calls"]
    return rec


def iter_tool_agent_events(
    registry: Any,
    user_query: str,
    *,
    max_turns: int = 8,
):
    """
    Yields streaming events for SSE: assistant / tool steps, then a final ``done`` payload.

    Each event is a dict with ``event`` in ``{"assistant", "tool", "done"}``.
    ``done`` matches the return shape of ``run_tool_agent`` plus ``"event": "done"``.
    """
    endpoint, model, timeout_sec = _ollama_config()
    trace: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_query.strip()},
    ]
    last_recommendations: Optional[List[Dict[str, Any]]] = None
    final_message = ""
    try:
        pseudo_max = int(os.environ.get("MOVIEMIND_AGENT_PSEUDO_TOOL_RETRIES", "2"))
    except ValueError:
        pseudo_max = 2
    pseudo_max = max(0, min(pseudo_max, 5))

    for turn in range(max_turns):
        pseudo_attempts = 0
        msg: Dict[str, Any] = {}
        tool_calls: List[Dict[str, Any]] = []

        while True:
            body = _call_ollama_chat(endpoint, model, messages, timeout_sec)
            msg = body.get("message") or {}
            tool_calls = _normalize_tool_calls(msg.get("tool_calls"))
            content = str(msg.get("content") or "").strip()

            if tool_calls:
                trace.append({"turn": turn + 1, "assistant_message": msg})
                yield {
                    "event": "assistant",
                    "turn": turn + 1,
                    "assistant_message": msg,
                }
                messages.append(_assistant_record_from_message(msg))
                break

            if _content_looks_like_pseudo_tool_json(content) and pseudo_attempts < pseudo_max:
                trace.append(
                    {
                        "turn": turn + 1,
                        "assistant_message": msg,
                        "guardrail": "pseudo_tool_json_in_content_without_tool_calls",
                    }
                )
                yield {
                    "event": "assistant",
                    "turn": turn + 1,
                    "assistant_message": msg,
                }
                messages.append(_assistant_record_from_message(msg))
                messages.append({"role": "user", "content": _PSEUDO_TOOL_JSON_NUDGE})
                pseudo_attempts += 1
                continue

            trace.append({"turn": turn + 1, "assistant_message": msg})
            yield {
                "event": "assistant",
                "turn": turn + 1,
                "assistant_message": msg,
            }
            messages.append(_assistant_record_from_message(msg))
            final_message = content
            yield {
                "event": "done",
                "final_message": final_message,
                "recommendations": last_recommendations,
                "trace": trace,
                "turns_used": turn + 1,
                "model": model,
                "error": None,
            }
            return

        for tc in tool_calls:
            tname = tc["name"]
            tout, trecs = _execute_tool(registry, tname, tc.get("arguments") or {})
            if trecs is not None:
                last_recommendations = trecs
            trace.append({"turn": turn + 1, "tool": tname, "tool_output_preview": tout[:2000]})
            yield {
                "event": "tool",
                "turn": turn + 1,
                "tool": tname,
                "tool_output_preview": tout[:2000],
            }
            _append_tool_message(messages, tname, tout)

    yield {
        "event": "done",
        "final_message": final_message or "Stopped after max_turns without a final reply.",
        "recommendations": last_recommendations,
        "trace": trace,
        "turns_used": max_turns,
        "model": model,
        "error": "max_turns_exceeded",
    }


def run_tool_agent(registry: Any, user_query: str, *, max_turns: int = 8) -> Dict[str, Any]:
    """
    Run multi-turn Ollama chat with tool execution.

    Returns dict: final_message, recommendations (last rec list from tools), trace, turns_used, error (optional).
    """
    for ev in iter_tool_agent_events(registry, user_query, max_turns=max_turns):
        if ev.get("event") == "done":
            out = {k: v for k, v in ev.items() if k != "event"}
            return out
    raise RuntimeError("tool agent finished without a terminal event")
