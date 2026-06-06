import unittest

from src.api.agent_loop import (
    _accumulate_recommendations,
    _extract_pseudo_tools_from_content,
    _merge_tool_calls,
    _queue_overflow_tools,
    query_wants_opposite_taste,
)
from src.api.genre_helpers import (
    genre_tokens_from_exclude_args,
    movie_excludes_genre_tokens,
    movie_matches_genre_tokens,
)


class TestOppositeTasteHelpers(unittest.TestCase):
    def test_query_wants_opposite_taste(self):
        self.assertTrue(query_wants_opposite_taste("top 10 movies opposite to what I usually like"))
        self.assertTrue(query_wants_opposite_taste("something unlike my usual taste"))
        self.assertFalse(query_wants_opposite_taste("top 10 action movies"))
        self.assertFalse(query_wants_opposite_taste("diverse genres please"))

    def test_genre_exclude_tokens(self):
        expanded, unknown = genre_tokens_from_exclude_args({"genre_exclude": ["Action", "sci-fi"]})
        self.assertEqual(expanded, ["action", "sci-fi"])
        self.assertEqual(unknown, [])

    def test_movie_excludes_genre_tokens(self):
        self.assertTrue(movie_excludes_genre_tokens("Comedy|Romance", ["action", "drama"]))
        self.assertFalse(movie_excludes_genre_tokens("Action|Sci-Fi", ["action"]))
        self.assertTrue(movie_matches_genre_tokens("Action|Sci-Fi", ["action"]))

    def test_extract_pseudo_tools_from_content(self):
        content = (
            '**Action:** {"name": "get_user_summary", "parameters": {"user_id": 1161}} '
            'and {"name": "get_recommendations", "parameters": {"model_id": "gradient_boosting", '
            '"user_id": 1161, "top_n": 12, "genre_any": "horror,mystery", "diversity_alpha": 0.25}}'
        )
        tools = _extract_pseudo_tools_from_content(content)
        names = [t["name"] for t in tools]
        self.assertEqual(names, ["get_user_summary", "get_recommendations"])
        self.assertEqual(tools[0]["arguments"]["user_id"], 1161)
        self.assertEqual(tools[1]["arguments"]["model_id"], "gradient_boosting")
        self.assertEqual(tools[1]["arguments"]["genre_any"], ["horror", "mystery"])

    def test_merge_tool_calls_adds_missing_from_content(self):
        native = [{"name": "list_available_models", "arguments": {}}]
        content_tools = [{"name": "get_user_summary", "arguments": {"user_id": 1161}}]
        merged = _merge_tool_calls(native, content_tools)
        self.assertEqual([t["name"] for t in merged], ["list_available_models", "get_user_summary"])

    def test_queue_overflow_tools(self):
        tools = [
            {"name": "list_available_models", "arguments": {}},
            {"name": "get_user_summary", "arguments": {"user_id": 1}},
            {"name": "get_recommendations", "arguments": {"top_n": 5}},
        ]
        now, queued = _queue_overflow_tools(tools)
        self.assertEqual(len(now), 1)
        self.assertEqual(len(queued), 2)
        self.assertEqual(now[0]["name"], "list_available_models")

    def test_accumulate_recommendations_merges_batches(self):
        a = _accumulate_recommendations(None, [{"movie_id": 1, "title": "A"}], "horror")
        b = _accumulate_recommendations(a, [{"movie_id": 2, "title": "B"}], "mystery")
        self.assertEqual(len(b), 2)
        self.assertEqual(b[0]["recommendation_batch"], "horror")
        self.assertEqual(b[1]["recommendation_batch"], "mystery")


if __name__ == "__main__":
    unittest.main()
