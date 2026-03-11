<div align="center">

# 🎯 prompt-hunter

**Automated Prompt Discovery for Open-Vocabulary Object Detection**

*Stop guessing prompts. Let AI find the best ones for you.*

[![CI](https://github.com/alanpeng/prompt-hunter/actions/workflows/ci.yml/badge.svg)](https://github.com/alanpeng/prompt-hunter/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## The Problem

Open-vocabulary object detectors are incredibly powerful — but their performance is **highly sensitive to the text prompt** you use. A slight wording change can swing mAP by 10+ points. Today, finding good prompts means manual trial-and-error.

## The Solution

**prompt-hunter** automates this entirely. It builds a closed-loop optimization system:

```
┌───────────────────┐      ┌────────────────────┐      ┌─────────────────────┐
│   Instance Crops  │─────▶│  Vision-Language    │─────▶│   Grounding Model   │
│                   │      │  Model (VLM)        │      │   Evaluator         │
│  From your COCO   │      │                     │      │                     │
│  annotations      │      │  Generates diverse  │      │  Scores each prompt │
│                   │      │  prompt candidates  │      │  on validation data │
└───────────────────┘      └────────────────────┘      └──────────┬──────────┘
                                                                  │
                                                                  ▼
                                                       ┌─────────────────────┐
                                                       │   Ranked Results    │
                                                       │                     │
                                                       │  "yellow forklift"  │
                                                       │   success: 95%      │
                                                       │   confidence: 0.87  │
                                                       └─────────────────────┘
```

## Why Use This?

| Without prompt-hunter | With prompt-hunter |
|---|---|
| Manual prompt guessing | Automated discovery |
| Single prompt attempt | Dozens of diverse candidates |
| "Did it work?" uncertainty | Quantified success rate, avg confidence, P10 reliability |
| Hours of human iteration | One command, walk away |

## Installation

```bash
pip install prompt-hunter
```

**From source:**
```bash
git clone https://github.com/alanpeng/prompt-hunter.git
cd prompt-hunter
pip install -e ".[dev]"
```

**Docker (recommended for complex GPU setups):**
```bash
docker build -t prompt-hunter .
docker run --gpus all -v /path/to/data:/data prompt-hunter \
    --class forklift --train-json /data/train.json --train-images /data/train/
```

## Quickstart

### CLI

```bash
# Basic — discover prompts for "person" class
prompt-hunter \
    --class person \
    --train-json data/train_annotations.json \
    --train-images data/train/ \
    --output results.txt

# Advanced — separate val set, more mining crops, custom models
prompt-hunter \
    --class forklift \
    --train-json data/train.json \
    --train-images data/train/ \
    --val-json data/val.json \
    --val-images data/val/ \
    --mining-limit 30 \
    --eval-limit 100 \
    --top-k 5 \
    --verbose
```

**Expected output:**
```
============================================================
  PROMPT HUNTING RESULTS — class: 'forklift'
============================================================

  Rank 1:
    Prompt:           'yellow forklift truck'
    Success Rate:     95.0%
    Avg Confidence:   0.8723
    P10 Confidence:   0.7201
    Min Confidence:   0.6104
  ----------------------------------------
  Rank 2:
    Prompt:           'industrial forklift'
    Success Rate:     90.0%
    Avg Confidence:   0.8156
    ...
```

### Python API

```python
from prompt_hunter import PromptHunter

hunter = PromptHunter()
results = hunter.hunt(
    annotation_path="data/train.json",
    image_root="data/train/",
    target_class="forklift",
    val_annotation_path="data/val.json",  # optional
    val_image_root="data/val/",           # optional
)

for r in results:
    print(f"{r.prompt!r}  success={r.success_rate:.0%}  conf={r.avg_confidence:.3f}")
```

### Using Individual Components

Each stage is independently usable:

```python
from prompt_hunter import InstanceCropper, PromptMiner, PromptEvaluator

# Stage 1: Crop instances
cropper = InstanceCropper("annotations.json", "images/")
crops = cropper.crop("vehicle", limit=20)

# Stage 2: Generate prompt candidates
miner = PromptMiner()
candidates = miner.generate(crops, repeats=2)
# → ['yellow forklift', 'warehouse vehicle', 'industrial truck', ...]

# Stage 3: Evaluate and rank
evaluator = PromptEvaluator()
results = evaluator.evaluate(val_crops, candidates)
# → [EvaluationResult(prompt='yellow forklift', success_rate=0.95, ...), ...]
```

## Ranking Metrics

Candidates are ranked by a **composite lexicographic sort**:

| Metric | Formula | Purpose |
|---|---|---|
| **Success Rate** ↓ | `mean(s_i > τ)` | Primary: what fraction of images got a detection? |
| **Avg Confidence** ↓ | `mean(s_i)` | Secondary: how confident were the detections? |
| **P10 Confidence** ↓ | `percentile(s_i, 10)` | Tiebreaker: how bad is the worst 10%? |

Where `s_i` is the best detection confidence on image `i`, and `τ = 0.5` is the failure threshold.

## Configuration

All behaviour is controlled via dataclass configs:

```python
from prompt_hunter.miner import MinerConfig
from prompt_hunter.evaluator import EvaluatorConfig
from prompt_hunter.hunter import HuntConfig, PromptHunter

hunter = PromptHunter(config=HuntConfig(
    mining_limit=30,
    eval_limit=100,
    repeats=2,
    miner=MinerConfig(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        temperature=0.9,
        top_p=0.95,
    ),
    evaluator=EvaluatorConfig(
        model_id="IDEA-Research/grounding-dino-base",
        detection_threshold=0.5,
        top_k_results=5,
    ),
))
```

## Project Structure

```
prompt-hunter/
├── src/prompt_hunter/
│   ├── __init__.py      # Public API exports
│   ├── cli.py           # Command-line interface
│   ├── cropper.py       # COCO instance cropping
│   ├── evaluator.py     # Grounding-model prompt scoring
│   ├── hunter.py        # End-to-end pipeline orchestrator
│   └── miner.py         # VLM-based prompt generation
├── tests/               # Unit tests (pytest)
├── pyproject.toml       # Package config + tool settings
├── Dockerfile           # GPU-ready container
└── .github/workflows/   # CI pipeline
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- GPU recommended (runs on CPU, but slowly)
- ~8 GB VRAM for default VLM + grounding model

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[MIT](LICENSE)
