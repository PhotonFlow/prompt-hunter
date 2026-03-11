"""Unit tests for prompt_hunter.evaluator (offline, no model loading)."""

from __future__ import annotations

import numpy as np
import pytest

from prompt_hunter.evaluator import EvaluationResult, PromptEvaluator


class TestAggregation:
    """Tests for the _aggregate ranking logic (no GPU required)."""

    def test_ranking_by_success_rate(self) -> None:
        """Higher success rate should rank first."""
        scores = {
            "prompt_a": [0.9, 0.9, 0.1],  # 2/3 above 0.5
            "prompt_b": [0.9, 0.9, 0.9],  # 3/3 above 0.5
        }
        evaluator = PromptEvaluator.__new__(PromptEvaluator)
        evaluator.config = type(
            "C", (), {"failure_threshold": 0.5, "top_k_results": 10}
        )()

        results = evaluator._aggregate(scores)
        assert results[0].prompt == "prompt_b"
        assert results[1].prompt == "prompt_a"

    def test_tiebreak_by_avg_confidence(self) -> None:
        """Same success rate → higher avg confidence ranks first."""
        scores = {
            "low_conf": [0.6, 0.6, 0.6],
            "high_conf": [0.9, 0.9, 0.9],
        }
        evaluator = PromptEvaluator.__new__(PromptEvaluator)
        evaluator.config = type(
            "C", (), {"failure_threshold": 0.5, "top_k_results": 10}
        )()

        results = evaluator._aggregate(scores)
        assert results[0].prompt == "high_conf"

    def test_p10_calculation(self) -> None:
        """P10 should reflect the 10th percentile of scores."""
        # 50 zeros + 50 ones: P10 should be 0.0
        scores = {"prompt": [0.0] * 50 + [1.0] * 50}
        evaluator = PromptEvaluator.__new__(PromptEvaluator)
        evaluator.config = type(
            "C", (), {"failure_threshold": 0.5, "top_k_results": 10}
        )()

        results = evaluator._aggregate(scores)
        assert results[0].p10_confidence == pytest.approx(0.0, abs=0.01)

    def test_empty_scores(self) -> None:
        """Prompt with no scores should be excluded."""
        scores = {"empty": []}
        evaluator = PromptEvaluator.__new__(PromptEvaluator)
        evaluator.config = type(
            "C", (), {"failure_threshold": 0.5, "top_k_results": 10}
        )()

        results = evaluator._aggregate(scores)
        assert len(results) == 0


class TestEvaluationResultDataclass:
    """Tests for the EvaluationResult data container."""

    def test_fields(self) -> None:
        r = EvaluationResult(
            prompt="test",
            success_rate=0.8,
            avg_confidence=0.75,
            p10_confidence=0.5,
            min_confidence=0.1,
        )
        assert r.prompt == "test"
        assert r.success_rate == 0.8
