"""High-level orchestrator for the prompt hunting pipeline.

Combines :class:`InstanceCropper`, :class:`PromptMiner`, and
:class:`PromptEvaluator` into a single ``hunt()`` call that takes raw
annotation files and returns ranked prompt results.

Pipeline overview
-----------------

.. code-block:: text

    ┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
    │  InstanceCropper │────▶│  PromptMiner │────▶│PromptEvaluator │
    │                 │     │              │     │                │
    │ COCO JSON +     │     │ VLM ×        │     │ Grounding      │
    │ Images → Crops  │     │ Questions →  │     │ Model ×        │
    │                 │     │ Candidates   │     │ Candidates →   │
    │                 │     │              │     │ Ranked Results  │
    └─────────────────┘     └──────────────┘     └────────────────┘

The pipeline is designed so that each stage can also be used
independently via the public API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from prompt_hunter.cropper import InstanceCropper
from prompt_hunter.evaluator import EvaluationResult, EvaluatorConfig, PromptEvaluator
from prompt_hunter.miner import MinerConfig, PromptMiner

logger = logging.getLogger(__name__)


@dataclass
class HuntConfig:
    """End-to-end configuration for a prompt hunting run.

    Parameters
    ----------
    mining_limit : int
        Number of training crops to feed the VLM for candidate generation.
    eval_limit : int
        Number of validation crops to use when scoring candidates.
    repeats : int
        Number of question cycles per mining crop (more cycles → more
        diverse candidates, at the cost of VLM inference time).
    miner : MinerConfig
        Configuration for the VLM-based prompt miner.
    evaluator : EvaluatorConfig
        Configuration for the grounding-model evaluator.
    """

    mining_limit: int = 20
    eval_limit: int = 50
    repeats: int = 1
    miner: MinerConfig = MinerConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()


class PromptHunter:
    """End-to-end prompt discovery for open-vocabulary object detection.

    Orchestrates the three-stage pipeline:

    1. **Crop** — extract object instances from annotated images.
    2. **Mine** — feed crops to a VLM to generate diverse prompt candidates.
    3. **Evaluate** — score each candidate with a grounding model and rank
       by composite reliability metrics.

    Parameters
    ----------
    config : HuntConfig or None
        Pipeline-level configuration.  Defaults to ``HuntConfig()``.

    Example
    -------
    >>> hunter = PromptHunter()
    >>> results = hunter.hunt(
    ...     annotation_path="train.json",
    ...     image_root="images/train",
    ...     target_class="pallet",
    ... )
    >>> for r in results:
    ...     print(f"{r.prompt!r}  {r.success_rate:.0%}")

    Notes
    -----
    Both the VLM and the grounding model are loaded lazily on first call
    to ``hunt()``.  This allows inspection of configuration before
    committing GPU memory.
    """

    def __init__(self, config: HuntConfig | None = None) -> None:
        self.config = config or HuntConfig()
        self._miner: PromptMiner | None = None
        self._evaluator: PromptEvaluator | None = None

    def hunt(
        self,
        annotation_path: str | Path,
        image_root: str | Path,
        target_class: str,
        *,
        val_annotation_path: str | Path | None = None,
        val_image_root: str | Path | None = None,
    ) -> list[EvaluationResult]:
        """Run the full prompt hunting pipeline.

        Parameters
        ----------
        annotation_path : str or Path
            COCO JSON for the training split (used for mining).
        image_root : str or Path
            Image directory for the training split.
        target_class : str
            Object category to hunt prompts for.
        val_annotation_path : str or Path or None
            COCO JSON for the validation split.  If *None*, falls back
            to *annotation_path*.
        val_image_root : str or Path or None
            Image directory for validation.  If *None*, falls back to
            *image_root*.

        Returns
        -------
        list of EvaluationResult
            Ranked prompts, best first.
        """
        logger.info("Starting hunt for class '%s'", target_class)

        # --- Stage 1: Crop ---
        train_cropper = InstanceCropper(annotation_path, image_root)
        mining_crops = train_cropper.crop(
            target_class,
            limit=self.config.mining_limit,
            mode="mining",
        )
        logger.info("Stage 1 — cropped %d mining instances", len(mining_crops))

        if not mining_crops:
            logger.error("No crops found for class '%s'", target_class)
            return []

        # --- Stage 2: Mine ---
        miner = self._get_miner()
        candidates = miner.generate(mining_crops, repeats=self.config.repeats)
        logger.info("Stage 2 — generated %d candidates", len(candidates))

        if not candidates:
            logger.error("VLM produced no usable candidates")
            return []

        # --- Stage 3: Evaluate ---
        val_ann = val_annotation_path or annotation_path
        val_root = val_image_root or image_root

        val_cropper = InstanceCropper(val_ann, val_root)
        eval_crops = val_cropper.crop(
            target_class,
            limit=self.config.eval_limit,
            mode="evaluation",
        )

        if not eval_crops:
            logger.warning("No validation crops; falling back to mining crops")
            eval_crops = mining_crops

        evaluator = self._get_evaluator()
        results = evaluator.evaluate(eval_crops, candidates)
        logger.info(
            "Stage 3 — evaluated %d candidates on %d images",
            len(candidates),
            len(eval_crops),
        )

        return results

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_miner(self) -> PromptMiner:
        if self._miner is None:
            self._miner = PromptMiner(config=self.config.miner)
        return self._miner

    def _get_evaluator(self) -> PromptEvaluator:
        if self._evaluator is None:
            self._evaluator = PromptEvaluator(config=self.config.evaluator)
        return self._evaluator
