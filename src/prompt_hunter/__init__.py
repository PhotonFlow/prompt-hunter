"""prompt-hunter: Automated prompt discovery for open-vocabulary object detection.

Discover optimal text prompts for zero-shot object detectors by using a
vision-language model to generate candidates, then a grounding model to
evaluate and rank them — replacing manual prompt engineering with a
closed-loop optimization system.

Typical usage::

    from prompt_hunter import PromptHunter

    hunter = PromptHunter()
    results = hunter.hunt(
        annotation_path="data/coco_annotations.json",
        image_root="data/images",
        target_class="forklift",
    )
    for r in results:
        print(f"{r.prompt!r}  success_rate={r.success_rate:.2%}")
"""

from __future__ import annotations

from prompt_hunter.cropper import InstanceCropper
from prompt_hunter.miner import PromptMiner
from prompt_hunter.evaluator import PromptEvaluator
from prompt_hunter.hunter import PromptHunter

__all__ = [
    "InstanceCropper",
    "PromptMiner",
    "PromptEvaluator",
    "PromptHunter",
]

__version__ = "0.1.0"
