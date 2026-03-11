"""Command-line interface for prompt-hunter.

Usage
-----
.. code-block:: bash

    # Basic usage
    prompt-hunter --class forklift \\
        --train-json data/train.json \\
        --train-images data/train/

    # With separate validation set and custom limits
    prompt-hunter --class person \\
        --train-json data/train.json \\
        --train-images data/train/ \\
        --val-json data/val.json \\
        --val-images data/val/ \\
        --mining-limit 30 \\
        --eval-limit 100 \\
        --output results/prompts_person.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from prompt_hunter.evaluator import EvaluatorConfig
from prompt_hunter.hunter import HuntConfig, PromptHunter
from prompt_hunter.miner import MinerConfig


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="prompt-hunter",
        description=(
            "Automatically discover optimal text prompts for "
            "open-vocabulary object detectors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  prompt-hunter --class forklift "
            "--train-json data/train.json --train-images data/train/\n"
            "  prompt-hunter --class person "
            "--train-json data/train.json --train-images data/train/ "
            "--val-json data/val.json --val-images data/val/"
        ),
    )

    # Required
    parser.add_argument(
        "--class",
        dest="target_class",
        type=str,
        required=True,
        help="Target object class to find prompts for.",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        required=True,
        help="Path to COCO-format training annotation JSON.",
    )
    parser.add_argument(
        "--train-images",
        type=str,
        required=True,
        help="Directory containing training images.",
    )

    # Optional validation set
    parser.add_argument(
        "--val-json",
        type=str,
        default=None,
        help="Path to validation JSON. Defaults to --train-json.",
    )
    parser.add_argument(
        "--val-images",
        type=str,
        default=None,
        help="Directory containing validation images. Defaults to --train-images.",
    )

    # Pipeline controls
    parser.add_argument(
        "--mining-limit",
        type=int,
        default=20,
        help="Number of crops for VLM prompt generation (default: 20).",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=50,
        help="Number of crops for prompt evaluation (default: 50).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Question cycles per mining crop (default: 1).",
    )

    # Model overrides
    parser.add_argument(
        "--vlm-model",
        type=str,
        default=MinerConfig.model_id,
        help=f"VLM model ID (default: {MinerConfig.model_id}).",
    )
    parser.add_argument(
        "--grounding-model",
        type=str,
        default=EvaluatorConfig.model_id,
        help=f"Grounding model ID (default: {EvaluatorConfig.model_id}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to return (default: 10).",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results. If not set, prints to stdout.",
    )

    # Misc
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build configs
    miner_cfg = MinerConfig(model_id=args.vlm_model)
    eval_cfg = EvaluatorConfig(
        model_id=args.grounding_model,
        top_k_results=args.top_k,
    )
    hunt_cfg = HuntConfig(
        mining_limit=args.mining_limit,
        eval_limit=args.eval_limit,
        repeats=args.repeats,
        miner=miner_cfg,
        evaluator=eval_cfg,
    )

    # Run
    hunter = PromptHunter(config=hunt_cfg)
    results = hunter.hunt(
        annotation_path=args.train_json,
        image_root=args.train_images,
        target_class=args.target_class,
        val_annotation_path=args.val_json,
        val_image_root=args.val_images,
    )

    # Format output
    lines = [
        f"{'='*60}",
        f"  PROMPT HUNTING RESULTS — class: {args.target_class!r}",
        f"{'='*60}",
        "",
    ]
    for rank, r in enumerate(results, start=1):
        lines.append(f"  Rank {rank}:")
        lines.append(f"    Prompt:           {r.prompt!r}")
        lines.append(f"    Success Rate:     {r.success_rate:.1%}")
        lines.append(f"    Avg Confidence:   {r.avg_confidence:.4f}")
        lines.append(f"    P10 Confidence:   {r.p10_confidence:.4f}")
        lines.append(f"    Min Confidence:   {r.min_confidence:.4f}")
        lines.append(f"  {'-'*40}")

    output_text = "\n".join(lines)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Results saved to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
