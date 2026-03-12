"""Prompt evaluation via grounding-model object detection.

Scores candidate text prompts by measuring how reliably an open-vocabulary
object detector can localise objects when guided by each prompt.  Each
candidate is ranked by a composite metric: (success_rate, avg_confidence,
p10_confidence).

Evaluation metrics
------------------
For each candidate prompt *p* and a set of validation crops
{x₁, …, xₙ}, we run the grounding model and record the best detection
confidence *s_i* (or 0 if nothing is detected above threshold):

- **Success rate** — fraction of crops with s_i > τ_fail
- **Average confidence** — mean(s₁, …, sₙ)
- **P10 confidence** — 10th-percentile of {s_i}, a robustness measure
  that penalises prompts with long left tails of low-confidence detections.

Candidates are ranked lexicographically by (success_rate ↓, avg_conf ↓,
p10_conf ↓).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Evaluation metrics for a single prompt candidate.

    Attributes
    ----------
    prompt : str
        The evaluated text prompt.
    success_rate : float
        Fraction of validation images where the detector found at least
        one object above the failure threshold.
    avg_confidence : float
        Mean best-detection confidence across all validation images.
    p10_confidence : float
        10th percentile of best-detection confidences — measures
        worst-case reliability.
    min_confidence : float
        Absolute minimum confidence observed.
    """

    prompt: str
    success_rate: float
    avg_confidence: float
    p10_confidence: float
    min_confidence: float


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluatorConfig:
    """Knobs for prompt evaluation.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for the grounding model.
    detection_threshold : float
        Minimum confidence for a detection to be considered valid.
    text_threshold : float
        Minimum text-grounding score for token–box associations.
    failure_threshold : float
        Confidence below which a detection is counted as a "failure"
        when computing *success_rate*.  Set slightly below
        *detection_threshold* to be lenient about borderline cases.
    top_k_results : int
        Number of top-ranked candidates to return.
    """

    model_id: str = "IDEA-Research/grounding-dino-base"
    detection_threshold: float = 0.6
    text_threshold: float = 0.6
    failure_threshold: float = 0.5
    top_k_results: int = 10


# ------------------------------------------------------------------
# Core class
# ------------------------------------------------------------------


class PromptEvaluator:
    """Scores prompt candidates using an open-vocabulary grounding model.

    For each candidate prompt, the evaluator runs detection on every
    validation crop and aggregates per-prompt statistics.  The final
    ranking uses a composite sort key to surface prompts that are both
    reliable (high success rate) and confident (high average score).

    Parameters
    ----------
    config : EvaluatorConfig or None
        Evaluation hyper-parameters.
    device : str or None
        PyTorch device.  Auto-detected if *None*.

    Example
    -------
    >>> evaluator = PromptEvaluator()
    >>> results = evaluator.evaluate(
    ...     ["val/001.jpg", "val/002.jpg"],
    ...     ["yellow forklift", "industrial vehicle"],
    ... )
    >>> print(results[0].prompt, results[0].success_rate)
    """

    def __init__(
        self,
        config: EvaluatorConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or EvaluatorConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(
            "Loading grounding model '%s' on %s",
            self.config.model_id,
            self.device,
        )
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.model_id
        ).to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        crop_paths: Sequence[str | Path],
        candidates: Sequence[str],
    ) -> list[EvaluationResult]:
        """Score and rank prompt candidates against validation crops.

        Parameters
        ----------
        crop_paths : sequence of str or Path
            Paths to validation image crops.
        candidates : sequence of str
            Candidate prompt strings to evaluate.

        Returns
        -------
        list of EvaluationResult
            Ranked results (best first), limited to
            ``config.top_k_results`` entries.
        """
        # Accumulate per-prompt score arrays
        scores: dict[str, list[float]] = {p: [] for p in candidates}

        for idx, img_path in enumerate(crop_paths):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                logger.warning("Skipping unreadable image: %s", img_path)
                continue

            for prompt in candidates:
                best = self._detect_best_score(image, prompt)
                scores[prompt].append(best)

            if (idx + 1) % 10 == 0:
                logger.info("Evaluated %d/%d images", idx + 1, len(crop_paths))

        # Aggregate and rank
        results = self._aggregate(scores)
        return results[: self.config.top_k_results]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_best_score(self, image: Image.Image, prompt: str) -> float:
        """Run detection and return the best confidence score (or 0)."""
        clean = prompt.strip().lower()
        if not clean.endswith("."):
            clean += "."

        inputs = self._processor(
            images=[image],
            text=clean,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.detection_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        det_scores = results.get("scores")
        if det_scores is not None and len(det_scores) > 0:
            return float(det_scores.max().item())
        return 0.0

    def _aggregate(self, scores: dict[str, list[float]]) -> list[EvaluationResult]:
        """Convert raw per-image scores into ranked EvaluationResults."""
        results: list[EvaluationResult] = []

        for prompt, vals in scores.items():
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                continue

            result = EvaluationResult(
                prompt=prompt,
                success_rate=float(np.mean(arr > self.config.failure_threshold)),
                avg_confidence=float(arr.mean()),
                p10_confidence=float(np.percentile(arr, 10)),
                min_confidence=float(arr.min()),
            )
            results.append(result)

        # Lexicographic sort: success_rate ↓, avg_conf ↓, p10 ↓
        results.sort(
            key=lambda r: (r.success_rate, r.avg_confidence, r.p10_confidence),
            reverse=True,
        )
        return results
