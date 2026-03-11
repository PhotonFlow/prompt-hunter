"""Example: Discover optimal prompts for a target class.

This example shows the end-to-end prompt hunting workflow using the
Python API.  Replace the paths below with your own COCO-format dataset.

Usage:
    python examples/basic_usage.py
"""

from prompt_hunter import PromptHunter
from prompt_hunter.hunter import HuntConfig
from prompt_hunter.miner import MinerConfig
from prompt_hunter.evaluator import EvaluatorConfig


def main() -> None:
    # Configure the pipeline
    config = HuntConfig(
        mining_limit=20,    # Number of crops for VLM mining
        eval_limit=50,      # Number of crops for evaluation
        repeats=1,          # Question cycles per crop
        miner=MinerConfig(
            temperature=0.8,
            top_p=0.95,
        ),
        evaluator=EvaluatorConfig(
            detection_threshold=0.6,
            top_k_results=5,
        ),
    )

    hunter = PromptHunter(config=config)

    # Run the hunt
    results = hunter.hunt(
        annotation_path="data/train_annotations.json",
        image_root="data/train/",
        target_class="forklift",
        val_annotation_path="data/val_annotations.json",
        val_image_root="data/val/",
    )

    # Display results
    print(f"\n{'='*60}")
    print(f"  Found {len(results)} ranked prompts for 'forklift'")
    print(f"{'='*60}\n")

    for rank, r in enumerate(results, 1):
        print(f"  #{rank}  {r.prompt!r}")
        print(f"       Success Rate:   {r.success_rate:.1%}")
        print(f"       Avg Confidence: {r.avg_confidence:.4f}")
        print(f"       P10 Confidence: {r.p10_confidence:.4f}")
        print()


if __name__ == "__main__":
    main()
