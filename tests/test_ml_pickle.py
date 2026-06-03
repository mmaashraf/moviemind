"""Smoke tests for sklearn model serialization used by ml_models / API."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ml_models import save_sklearn_estimator  # noqa: E402


class TestSklearnPickle(unittest.TestCase):
    def test_save_sklearn_estimator_roundtrip_joblib_load(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(80, 7)
        y = rng.randn(80)
        model = GradientBoostingRegressor(
            n_estimators=12, max_depth=3, learning_rate=0.1, random_state=0
        )
        model.fit(X, y)

        tmp = tempfile.mkdtemp()
        path = save_sklearn_estimator(model, "gb_test.pkl", models_dir=tmp)
        self.assertTrue(os.path.isfile(path))

        loaded = joblib.load(path)
        np.testing.assert_allclose(loaded.predict(X[:10]), model.predict(X[:10]), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
