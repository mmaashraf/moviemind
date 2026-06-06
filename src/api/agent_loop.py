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
    genre_tokens_from_exclude_args,
    genre_tokens_from_tool_args,
    movie_excludes_genre_tokens,
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
                            "NOT for opposite/unlike taste — use genre_exclude instead."
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
                    "genre_exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional. Drop movies whose genres contain ANY of these tokens (after alias expansion). "
                            "Use for opposite/unlike taste: pass the user's top_genres_train from get_user_summary."
                        ),
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
- **Never ask the user clarifying questions.** Use session context (user_id, model_id, top_n) and tools. If data is missing, state the limit briefly and still return whatever tool results you have.
- **Language understanding** is encouraged: in your **Thought**, map the user’s casual words to **MovieLens genre tokens** (or to known alias words the server understands, e.g. `fiction` → narrative genres). You may use general knowledge of synonyms—e.g. “scary” → horror, “funny” → comedy, “space” → sci-fi—then pass the **canonical** tokens in `genre_filter` / `genre_any`. If a phrase has no good single tag, pick the closest genre(s) and say what you assumed.
- If the user wants data outside these tools (live web, non-MovieLens catalogs), say so in **Thought** and explain the limit.

## ReAct format (every assistant turn)
1. **Thought:** Start the `content` field with a short paragraph (1–4 sentences). Explain what you know, what you still need, and what tool call(s) you will make next—or that you are ready to give the final answer.
2. **Action:** When you need data, emit **tool_calls** (the chat API supports tools). One turn may include multiple tool calls if independent.
3. **Observation:** After each tool result appears in the conversation, continue in a new turn with another **Thought**, then more **Action** if needed.

Never leave `content` empty on a turn where you call tools—always lead with **Thought:** so the user can follow your reasoning.
Do **not** print lines like `{"name":"get_recommendations",...}` as your whole answer—those are **not** executed. Only native **`tool_calls`** run on the server.

## Tool usage hints
- When session context includes **user_id**, call **get_user_summary(user_id)** FIRST if the query mentions taste, preferences, “what I usually like”, opposite/unlike/contrarian requests, or genre intent.
- Unsure which checkpoints exist → **list_available_models** (respect `available`).
- Recommendations → **get_recommendations** with valid **model_id** and **user_id** from session context (do not invent IDs).
- **Opposite / unlike usual taste:** (1) **get_user_summary** → read **top_genres_train**; (2) **get_recommendations** with **genre_exclude** set to those top genres (array). Do **not** use **diversity_alpha** for this — diversity only spreads genres among high-scoring picks.
- **Diversity:** If the user wants **varied** genres among good picks, less repetition — set **`diversity_alpha`** (e.g. **0.15–0.35**). Omit or **0** for maximum score-only ranking.
- Genre include constraints → set **genre_any** or **genre_filter**. Request a generous **top_n** (e.g. 30–40) when filtering; the server also over-fetches when filters apply.
- User IDs must be within dataset bounds. If a tool reports an invalid `user_id`, explain the valid range in one sentence — do not ask the user to reply.

## MovieLens / registry reminders
Model IDs: baseline_global_mean, linear_regression, random_forest, gradient_boosting, ncf_baseline, ncf_tuned (some may be unavailable on disk).

Genres are fixed lowercase tokens: action, adventure, animation, children, documentary, drama, sci-fi, thriller, comedy, romance, crime, horror, mystery, western, war, fantasy, musical, film-noir, ...
There is NO tag literally named "fiction". Use **genre_any** / **genre_filter** with `documentary` and `fiction` (the backend expands `fiction` to narrative genres). Always pass genre parameters when the user asks for specific kinds of movies.

**Final answer:** When tools have given enough data, your last turn should be **Thought** (short recap) plus a concise list of movie titles from tool outputs. Keep it brief — the UI shows the recommendation table separately.

**Tool calling (critical):** Emit tools ONLY via native **`tool_calls`**. Do NOT print `{"name": "...", "parameters": {...}}` in `content` — the server may parse that as a fallback, but native tool_calls are required for reliable execution. Never invent user stats; use **get_user_summary** output only.
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
                "hint": "Use a user_id inside this range from session context.",
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
            exclude_expanded, exclude_unknown = genre_tokens_from_exclude_args(arguments)
            has_include = bool(expanded)
            has_exclude = bool(exclude_expanded)
            fetch_n = fetch_size_for_genre_filter(top_n, has_include or has_exclude)

            recs = registry.recommend(mid, uid, fetch_n, alpha)
            meta: Dict[str, Any] = {
                "requested_top_n": top_n,
                "scored_candidates": len(recs),
                "fetch_n": fetch_n,
                "genre_expanded_tokens": expanded,
                "genre_unknown_phrases": unknown_raw,
                "genre_exclude_tokens": exclude_expanded,
                "genre_exclude_unknown": exclude_unknown,
            }
            if has_include:
                recs = [r for r in recs if movie_matches_genre_tokens(r.get("genres"), expanded)]
                meta["after_genre_filter"] = len(recs)
                meta["note"] = (
                    "OR-match: movie kept if any expanded genre token appears in its genres pipe-string. "
                    "`fiction` expands to narrative genres (all except documentary)."
                )
            if has_exclude:
                recs = [r for r in recs if movie_excludes_genre_tokens(r.get("genres"), exclude_expanded)]
                meta["after_genre_exclude"] = len(recs)
                meta.setdefault(
                    "note",
                    "Exclude-match: movie dropped if any excluded genre token appears in its genres pipe-string.",
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


_OPPOSITE_TASTE_RE = re.compile(
    r"\b("
    r"opposite|opposing|contrary|contrarian|unlike|anti(?:[\s-]?recommend)?|"
    r"not what i (?:usually|normally|typically)|"
    r"different from (?:my|what i)|"
    r"avoid my (?:usual|top|favorite|favourite)|"
    r"away from my (?:usual|taste|preferences)"
    r")\b",
    re.IGNORECASE,
)


def query_wants_opposite_taste(query: str) -> bool:
    return bool(_OPPOSITE_TASTE_RE.search(query or ""))


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


def _extract_top_n_from_query(query: str, default: int = 10) -> int:
    m = re.search(r"\btop\s+(\d{1,3})\b", query or "", re.IGNORECASE)
    if m:
        try:
            return max(1, min(100, int(m.group(1))))
        except ValueError:
            pass
    m2 = re.search(r"\b(\d{1,2})\s+movies?\b", query or "", re.IGNORECASE)
    if m2:
        try:
            return max(1, min(100, int(m2.group(1))))
        except ValueError:
            pass
    return default


def _build_agent_user_message(
    user_query: str,
    *,
    user_id: Optional[int] = None,
    model_id: Optional[str] = None,
    top_n: Optional[int] = None,
    prefetched_summary: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []
    ctx_lines: List[str] = []
    if user_id is not None:
        ctx_lines.append(f"- user_id: {int(user_id)}")
    if model_id:
        ctx_lines.append(f"- model_id: {str(model_id).strip()}")
    if top_n is not None:
        ctx_lines.append(f"- top_n: {max(1, min(100, int(top_n)))}")
    if ctx_lines:
        parts.append("## Session context (always use these; never ask the user for them)\n" + "\n".join(ctx_lines))
    if prefetched_summary is not None:
        parts.append(
            "## Prefetched get_user_summary (ground truth — do not invent conflicting stats)\n"
            + json.dumps(prefetched_summary, ensure_ascii=False, default=str)
        )
    parts.append("## User request\n" + user_query.strip())
    return "\n\n".join(parts)


_TOOL_NAME_IN_CONTENT_RE = _PSEUDO_TOOL_JSON_PATTERN

_TOOL_ARGS_IN_CONTENT_RE = re.compile(
    r'["\'](?:parameters|arguments)["\']\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
    re.IGNORECASE | re.DOTALL,
)

_TOOL_CALL_ORDER = {"list_available_models": 0, "get_user_summary": 1, "get_recommendations": 2}


def _coerce_tool_arguments(name: str, arguments: Any) -> Dict[str, Any]:
    args: Dict[str, Any] = dict(arguments) if isinstance(arguments, dict) else {}
    if name == "list_available_models":
        return {}
    ga = args.get("genre_any")
    if isinstance(ga, str) and ga.strip():
        args["genre_any"] = [p.strip() for p in re.split(r"[,|/]", ga) if p.strip()]
    return args


def _apply_session_defaults_to_tool_args(
    name: str,
    arguments: Dict[str, Any],
    *,
    user_id: Optional[int],
    model_id: Optional[str],
    top_n: Optional[int],
) -> Dict[str, Any]:
    args = _coerce_tool_arguments(name, arguments)
    if name == "get_user_summary" and user_id is not None and "user_id" not in args:
        args["user_id"] = int(user_id)
    if name == "get_recommendations":
        if user_id is not None and "user_id" not in args:
            args["user_id"] = int(user_id)
        if model_id and not args.get("model_id"):
            args["model_id"] = str(model_id).strip()
        if top_n is not None and "top_n" not in args:
            args["top_n"] = max(1, min(100, int(top_n)))
    return args


def _extract_pseudo_tools_from_content(content: str) -> List[Dict[str, Any]]:
    """Parse tool name/parameter JSON blobs the model sometimes prints in plain content."""
    if not content or not _PSEUDO_TOOL_JSON_PATTERN.search(content):
        return []
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for name_match in _TOOL_NAME_IN_CONTENT_RE.finditer(content):
        name = str(name_match.group(1))
        tail = content[name_match.end() : name_match.end() + 800]
        args: Dict[str, Any] = {}
        args_match = _TOOL_ARGS_IN_CONTENT_RE.search(tail)
        if args_match:
            blob = args_match.group(1)
            try:
                parsed = json.loads(blob)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                pass
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "arguments": _coerce_tool_arguments(name, args)})
    out.sort(key=lambda t: _TOOL_CALL_ORDER.get(t["name"], 99))
    return out


def _merge_tool_calls(
    native: List[Dict[str, Any]],
    content_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    for tc in native + content_tools:
        name = tc.get("name")
        if not name:
            continue
        args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        merged.append({"name": str(name), "arguments": dict(args)})
    merged.sort(key=lambda t: _TOOL_CALL_ORDER.get(t["name"], 99))
    return merged


def _fallback_recommendations(
    registry: Any,
    *,
    user_id: Optional[int],
    model_id: Optional[str],
    top_n: Optional[int],
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Last-resort recommend when the LLM loop exhausts max_turns without recs."""
    if user_id is None or not model_id:
        return None, ""
    args = _apply_session_defaults_to_tool_args(
        "get_recommendations",
        {},
        user_id=user_id,
        model_id=model_id,
        top_n=top_n,
    )
    tout, recs = _execute_tool(registry, "get_recommendations", args)
    return recs, tout


def try_deterministic_opposite_recommendations(
    registry: Any,
    query: str,
    *,
    user_id: Optional[int],
    model_id: Optional[str],
    top_n: Optional[int],
) -> Optional[Dict[str, Any]]:
    """
    Handle opposite/unlike taste without LLM when UI passes user_id + model_id.
    """
    if user_id is None or not model_id or not query_wants_opposite_taste(query):
        return None

    uid = int(user_id)
    mid = str(model_id).strip()
    tn = int(top_n) if top_n is not None else _extract_top_n_from_query(query, 10)
    tn = max(1, min(100, tn))

    summary = registry.user_summary(uid)
    top_genres = list(summary.get("top_genres_train") or [])
    trace: List[Dict[str, Any]] = [
        {
            "turn": 1,
            "deterministic": "get_user_summary",
            "tool_output_preview": json.dumps(summary, ensure_ascii=False, default=str)[:2000],
        }
    ]

    if not top_genres:
        return None

    rec_args = {
        "model_id": mid,
        "user_id": uid,
        "top_n": tn,
        "diversity_alpha": 0.0,
        "genre_exclude": [str(g).strip().lower() for g in top_genres[:5] if str(g).strip()],
    }
    tout, recs = _execute_tool(registry, "get_recommendations", rec_args)
    trace.append(
        {
            "turn": 1,
            "deterministic": "get_recommendations",
            "tool": "get_recommendations",
            "tool_output_preview": tout[:2000],
        }
    )
    return {
        "final_message": "",
        "recommendations": recs or [],
        "trace": trace,
        "turns_used": 0,
        "model": "deterministic",
        "error": None if recs else "no_recommendations_after_exclude",
    }


def _recommendation_batch_label(arguments: Dict[str, Any]) -> str:
    ga = arguments.get("genre_any")
    if isinstance(ga, list) and ga:
        return ",".join(str(x) for x in ga[:3])
    gf = arguments.get("genre_filter") or arguments.get("genre")
    if isinstance(gf, str) and gf.strip():
        return gf.strip()
    return "general"


def _accumulate_recommendations(
    existing: Optional[List[Dict[str, Any]]],
    new: List[Dict[str, Any]],
    batch_label: str,
) -> List[Dict[str, Any]]:
    out = list(existing or [])
    seen = {int(r["movie_id"]) for r in out if r.get("movie_id") is not None}
    for row in new:
        item = dict(row)
        item["recommendation_batch"] = batch_label
        mid = item.get("movie_id")
        if mid is not None and int(mid) in seen:
            continue
        if mid is not None:
            seen.add(int(mid))
        out.append(item)
    return out


def _queue_overflow_tools(tool_calls: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """When the model plans multiple tools at once, run one now and queue the rest."""
    if len(tool_calls) <= 1:
        return tool_calls, []
    return tool_calls[:1], tool_calls[1:]


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
    user_id: Optional[int] = None,
    model_id: Optional[str] = None,
    top_n: Optional[int] = None,
    stop_after_recommendations: bool = True,
):
    """
    Yields streaming events for SSE: assistant / tool steps, then a final ``done`` payload.

    Each event is a dict with ``event`` in ``{"assistant", "tool", "done"}``.
    ``done`` matches the return shape of ``run_tool_agent`` plus ``"event": "done"``.
    """
    deterministic = try_deterministic_opposite_recommendations(
        registry,
        user_query,
        user_id=user_id,
        model_id=model_id,
        top_n=top_n,
    )
    if deterministic is not None:
        for entry in deterministic.get("trace") or []:
            tname = entry.get("tool") or entry.get("deterministic")
            if tname:
                yield {
                    "event": "tool",
                    "turn": entry.get("turn", 1),
                    "tool": tname,
                    "tool_output_preview": entry.get("tool_output_preview", ""),
                }
        yield {"event": "done", **deterministic}
        return

    endpoint, model, timeout_sec = _ollama_config()
    trace: List[Dict[str, Any]] = []
    prefetched_summary: Optional[Dict[str, Any]] = None
    if user_id is not None:
        try:
            prefetched_summary = registry.user_summary(int(user_id))
        except Exception:  # noqa: BLE001
            prefetched_summary = None
    user_content = _build_agent_user_message(
        user_query,
        user_id=user_id,
        model_id=model_id,
        top_n=top_n,
        prefetched_summary=prefetched_summary,
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    last_recommendations: Optional[List[Dict[str, Any]]] = None
    all_recommendations: List[Dict[str, Any]] = []
    tool_queue: List[Dict[str, Any]] = []
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
        from_queue = False

        if tool_queue:
            tc = tool_queue.pop(0)
            tc = {
                "name": tc["name"],
                "arguments": _apply_session_defaults_to_tool_args(
                    tc["name"],
                    tc.get("arguments") or {},
                    user_id=user_id,
                    model_id=model_id,
                    top_n=top_n,
                ),
            }
            tool_calls = [tc]
            from_queue = True
            msg = {
                "role": "assistant",
                "content": f"**Action:** `{tc['name']}` (queued step {turn + 1})",
            }
            trace.append({"turn": turn + 1, "assistant_message": msg, "note": "queued_tool_step"})
            yield {"event": "assistant", "turn": turn + 1, "assistant_message": msg}
            messages.append(_assistant_record_from_message(msg))
        else:
            while True:
                body = _call_ollama_chat(endpoint, model, messages, timeout_sec)
                msg = body.get("message") or {}
                tool_calls = _normalize_tool_calls(msg.get("tool_calls"))
                content = str(msg.get("content") or "").strip()
                content_tools = _extract_pseudo_tools_from_content(content)
                if content_tools:
                    tool_calls = _merge_tool_calls(tool_calls, content_tools)
                tool_calls = [
                    {
                        "name": tc["name"],
                        "arguments": _apply_session_defaults_to_tool_args(
                            tc["name"],
                            tc.get("arguments") or {},
                            user_id=user_id,
                            model_id=model_id,
                            top_n=top_n,
                        ),
                    }
                    for tc in tool_calls
                ]

                if tool_calls:
                    run_now, overflow = _queue_overflow_tools(tool_calls)
                    if overflow:
                        tool_queue.extend(overflow)
                    tool_calls = run_now
                    entry: Dict[str, Any] = {"turn": turn + 1, "assistant_message": msg}
                    notes: List[str] = []
                    if content_tools and not _normalize_tool_calls(msg.get("tool_calls")):
                        notes.append("executed_pseudo_tool_json_from_content")
                    if overflow:
                        notes.append(f"queued_{len(overflow)}_remaining_tools")
                    if notes:
                        entry["note"] = ";".join(notes)
                    trace.append(entry)
                    yield {"event": "assistant", "turn": turn + 1, "assistant_message": msg}
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
                    yield {"event": "assistant", "turn": turn + 1, "assistant_message": msg}
                    messages.append(_assistant_record_from_message(msg))
                    messages.append({"role": "user", "content": _PSEUDO_TOOL_JSON_NUDGE})
                    pseudo_attempts += 1
                    continue

                trace.append({"turn": turn + 1, "assistant_message": msg})
                yield {"event": "assistant", "turn": turn + 1, "assistant_message": msg}
                messages.append(_assistant_record_from_message(msg))
                final_message = content
                yield {
                    "event": "done",
                    "final_message": final_message,
                    "recommendations": last_recommendations or all_recommendations or None,
                    "trace": trace,
                    "turns_used": turn + 1,
                    "model": model,
                    "error": None,
                }
                return

        got_recs_this_turn = False
        for tc in tool_calls:
            tname = tc["name"]
            args = tc.get("arguments") or {}
            tout, trecs = _execute_tool(registry, tname, args)
            if trecs is not None:
                batch_label = _recommendation_batch_label(args)
                all_recommendations = _accumulate_recommendations(all_recommendations, trecs, batch_label)
                last_recommendations = all_recommendations
                got_recs_this_turn = True
            trace.append({"turn": turn + 1, "tool": tname, "tool_output_preview": tout[:2000]})
            yield {"event": "tool", "turn": turn + 1, "tool": tname, "tool_output_preview": tout[:2000]}
            _append_tool_message(messages, tname, tout)

        if stop_after_recommendations and got_recs_this_turn and not tool_queue:
            yield {
                "event": "done",
                "final_message": "",
                "recommendations": last_recommendations,
                "trace": trace,
                "turns_used": turn + 1,
                "model": model,
                "error": None,
            }
            return

    if last_recommendations is None and all_recommendations:
        last_recommendations = all_recommendations

    if last_recommendations is None and user_id is not None and model_id:
        recs, tout = _fallback_recommendations(
            registry,
            user_id=user_id,
            model_id=model_id,
            top_n=top_n,
        )
        if recs:
            last_recommendations = recs
            trace.append(
                {
                    "turn": max_turns,
                    "tool": "get_recommendations",
                    "tool_output_preview": tout[:2000],
                    "note": "server_fallback_after_max_turns",
                }
            )
            yield {
                "event": "done",
                "final_message": "",
                "recommendations": last_recommendations,
                "trace": trace,
                "turns_used": max_turns,
                "model": model,
                "error": None,
            }
            return

    yield {
        "event": "done",
        "final_message": final_message or "Stopped after max_turns without a final reply.",
        "recommendations": last_recommendations or all_recommendations or None,
        "trace": trace,
        "turns_used": max_turns,
        "model": model,
        "error": "max_turns_exceeded",
    }


def run_tool_agent(
    registry: Any,
    user_query: str,
    *,
    max_turns: int = 8,
    user_id: Optional[int] = None,
    model_id: Optional[str] = None,
    top_n: Optional[int] = None,
    stop_after_recommendations: bool = True,
) -> Dict[str, Any]:
    """
    Run multi-turn Ollama chat with tool execution.

    Returns dict: final_message, recommendations (last rec list from tools), trace, turns_used, error (optional).
    """
    for ev in iter_tool_agent_events(
        registry,
        user_query,
        max_turns=max_turns,
        user_id=user_id,
        model_id=model_id,
        top_n=top_n,
        stop_after_recommendations=stop_after_recommendations,
    ):
        if ev.get("event") == "done":
            out = {k: v for k, v in ev.items() if k != "event"}
            return out
    raise RuntimeError("tool agent finished without a terminal event")
