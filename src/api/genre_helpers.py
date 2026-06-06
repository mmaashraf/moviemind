"""
MovieLens 1M genre strings + user-facing aliases for agent/tool filtering.

Movie genres in data look like "Animation|Children's|Comedy" — tokens match
SUPPORTED_GENRES in nlp.py (lowercase, hyphenated).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from .nlp import SUPPORTED_GENRES

_CANON: Set[str] = set(SUPPORTED_GENRES)

# "fiction" is not a MovieLens label — treat as narrative (any genre except documentary).
_NARRATIVE_ONLY = [g for g in SUPPORTED_GENRES if g != "documentary"]

GENRE_ALIASES: Dict[str, List[str]] = {
    "fiction": list(_NARRATIVE_ONLY),
    "narrative": list(_NARRATIVE_ONLY),
    "movie": list(_NARRATIVE_ONLY),
    "nonfiction": ["documentary"],
    "non-fiction": ["documentary"],
    "non fiction": ["documentary"],
    "documentaries": ["documentary"],
    "sci-fi": ["sci-fi"],
    "science fiction": ["sci-fi"],
    "film noir": ["film-noir"],
    # Colloquial → MovieLens token(s); LLM can also map in Thought before calling tools.
    "scary": ["horror"],
    "funny": ["comedy"],
    "romantic": ["romance"],
    "kids": ["children"],
    "space": ["sci-fi"],
}


def split_genre_phrase(raw: str) -> List[str]:
    """Split comma/OR/slash-separated phrases."""
    s = raw.strip().lower()
    if not s:
        return []
    parts = re.split(r"\s*,\s*|\s*\|\s*|\s+(?:or)\s+|\s*/\s*", s)
    return [p.strip() for p in parts if p.strip()]


def expand_genre_token(token: str) -> List[str]:
    """Map one token to canonical MovieLens genre tokens."""
    t = token.strip().lower()
    if not t:
        return []
    if t in _CANON:
        return [t]
    if t in GENRE_ALIASES:
        return list(GENRE_ALIASES[t])
    rep = t.replace(" ", "-")
    if rep in _CANON:
        return [rep]
    return []


def genre_tokens_from_tool_args(arguments: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """
    Build expanded canonical genre tokens from get_recommendations arguments.

    Supports:
      - genre_any: ["documentary", "fiction"]
      - genre_filter / genre: "documentary,fiction" or "documentary or fiction"

    Returns (expanded_tokens, unknown_raw_chunks).
    """
    chunks: List[str] = []
    raw_any = arguments.get("genre_any")
    if isinstance(raw_any, list):
        for x in raw_any:
            s = str(x).strip()
            if s:
                chunks.append(s)
    combined = arguments.get("genre_filter") or arguments.get("genre")
    if isinstance(combined, str) and combined.strip():
        chunks.extend(split_genre_phrase(combined))

    expanded: List[str] = []
    unknown: List[str] = []
    seen: Set[str] = set()

    for chunk in chunks:
        ex = expand_genre_token(chunk)
        if ex:
            for tok in ex:
                if tok not in seen:
                    seen.add(tok)
                    expanded.append(tok)
            continue
        # multi-token chunk?
        subs = split_genre_phrase(chunk.replace(" and ", ", "))
        got_any = False
        for sub in subs:
            ex2 = expand_genre_token(sub)
            if ex2:
                got_any = True
                for tok in ex2:
                    if tok not in seen:
                        seen.add(tok)
                        expanded.append(tok)
            elif sub.strip():
                unknown.append(sub.strip().lower())
        if not got_any and chunk.strip():
            low = chunk.strip().lower()
            if low not in unknown:
                unknown.append(low)

    # Stable de-dupe unknown hints
    unknown = list(dict.fromkeys(unknown))
    return expanded, unknown


def movie_matches_genre_tokens(movie_genres: Any, expanded_tokens: List[str]) -> bool:
    """Row matches if any canonical token appears in the movie's genre pipe-string."""
    if not expanded_tokens:
        return True
    tags = {g.strip().lower() for g in str(movie_genres).split("|") if g.strip()}
    return bool(tags & set(expanded_tokens))


def genre_tokens_from_exclude_args(arguments: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """
    Build expanded canonical genre tokens from get_recommendations exclude arguments.

    Supports:
      - genre_exclude: ["action", "drama"] or comma-separated string
    """
    chunks: List[str] = []
    raw_exclude = arguments.get("genre_exclude")
    if isinstance(raw_exclude, list):
        for x in raw_exclude:
            s = str(x).strip()
            if s:
                chunks.append(s)
    elif isinstance(raw_exclude, str) and raw_exclude.strip():
        chunks.extend(split_genre_phrase(raw_exclude))

    expanded: List[str] = []
    unknown: List[str] = []
    seen: Set[str] = set()
    for chunk in chunks:
        ex = expand_genre_token(chunk)
        if ex:
            for tok in ex:
                if tok not in seen:
                    seen.add(tok)
                    expanded.append(tok)
            continue
        low = chunk.strip().lower()
        if low in _CANON and low not in seen:
            seen.add(low)
            expanded.append(low)
        elif low:
            unknown.append(low)
    unknown = list(dict.fromkeys(unknown))
    return expanded, unknown


def movie_excludes_genre_tokens(movie_genres: Any, excluded_tokens: List[str]) -> bool:
    """Row kept when none of the excluded canonical tokens appear in its genre pipe-string."""
    if not excluded_tokens:
        return True
    return not movie_matches_genre_tokens(movie_genres, excluded_tokens)


def fetch_size_for_genre_filter(requested_top_n: int, has_filter: bool) -> int:
    """Fetch extra scored rows before OR-filtering so enough survive."""
    if not has_filter:
        return max(1, min(100, requested_top_n))
    return min(300, max(requested_top_n * 25, 80))
