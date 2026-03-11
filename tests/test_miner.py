"""Unit tests for prompt_hunter.miner (post-processing only)."""

from __future__ import annotations

import pytest

from prompt_hunter.miner import PromptMiner


class TestCleanMethod:
    """Tests for PromptMiner._clean (no GPU required)."""

    def test_extracts_after_inst_tag(self) -> None:
        raw = "Some preamble [/INST] A yellow forklift."
        result = PromptMiner._clean(raw)
        assert result == "a yellow forklift"

    def test_removes_filler_phrases(self) -> None:
        raw = "[/INST] A photo of a warehouse truck."
        result = PromptMiner._clean(raw)
        assert result == "a warehouse truck"

    def test_strips_punctuation(self) -> None:
        raw = "[/INST] ...a pallet!!!"
        result = PromptMiner._clean(raw)
        assert result == "a pallet"

    def test_rejects_short_strings(self) -> None:
        raw = "[/INST] hi"
        result = PromptMiner._clean(raw)
        assert result is None

    def test_handles_no_inst_tag(self) -> None:
        raw = "Just a plain response about a box"
        result = PromptMiner._clean(raw)
        assert result is not None
        assert "box" in result

    def test_multiple_fillers(self) -> None:
        raw = "[/INST] This is an image of a small car."
        result = PromptMiner._clean(raw)
        assert result is not None
        assert "small car" in result
