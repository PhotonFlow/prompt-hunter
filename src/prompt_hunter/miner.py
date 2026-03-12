"""Vision-language model based prompt candidate generation.

Uses a multimodal language model to inspect cropped object instances and
generate diverse textual descriptions that can serve as prompts for
open-vocabulary object detectors.

The generation strategy asks the VLM multiple complementary questions per
image (naming, description, sentence completion) and uses nucleus sampling
with elevated temperature to maximise candidate diversity.

Mathematical context
--------------------
Given an image crop *x* and a question template *q_j*, the VLM generates a
response token sequence::

    ŷ = argmax_{y} Π_t  P(y_t | y_{<t}, x, q_j; θ)

With ``do_sample=True``, ``temperature=τ``, and ``top_p=p``, each token is
sampled from the truncated distribution after re-scaling logits by 1/τ and
retaining the minimal set whose cumulative probability ≥ *p*.  Higher τ
yields more diverse (less deterministic) completions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class MinerConfig:
    """Knobs for prompt candidate generation.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for the vision-language model.
    temperature : float
        Sampling temperature; higher → more diverse outputs.
        Default ``0.8`` balances creativity and coherence.
    top_p : float
        Nucleus sampling threshold.  Retains tokens whose cumulative
        probability mass reaches *top_p*.
    max_new_tokens : int
        Maximum number of tokens the VLM generates per question.
    questions : tuple of str
        Prompt templates posed to the VLM for each crop.  ``{image}``
        is replaced internally by the model's image token.
    """

    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 40
    questions: tuple[str, ...] = (
        "Describe this object in a short phrase for object detection.",
        "What is the specific name of this object?",
        "Describe the visual appearance of this object concisely.",
        "Complete this sentence: A photo of a ...",
    )


# ------------------------------------------------------------------
# Core class
# ------------------------------------------------------------------


class PromptMiner:
    """Generates text-prompt candidates from cropped object images.

    The miner feeds each crop through a vision-language model with
    multiple complementary questions.  Responses are post-processed
    (lowercased, stripped of filler phrases, de-duplicated) and returned
    as a list of unique candidate prompts.

    Parameters
    ----------
    config : MinerConfig or None
        Generation hyper-parameters.  Defaults to ``MinerConfig()``.
    device : str or None
        PyTorch device string.  Auto-detected if *None*.

    Example
    -------
    >>> miner = PromptMiner()
    >>> candidates = miner.generate(["crops/001.jpg", "crops/002.jpg"])
    >>> print(candidates)
    ['yellow forklift', 'industrial vehicle', 'warehouse truck']
    """

    _PROMPT_TEMPLATE = "[INST] <image>\n{question} [/INST]"
    _FILLER_PHRASES = ("a photo of", "an image of", "this is", "it is")

    def __init__(
        self,
        config: MinerConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or MinerConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("Loading VLM '%s' on %s", self.config.model_id, self.device)
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        self._processor = LlavaNextProcessor.from_pretrained(self.config.model_id)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        crop_paths: Sequence[str | Path],
        *,
        repeats: int = 1,
    ) -> list[str]:
        """Generate prompt candidates from a collection of image crops.

        Parameters
        ----------
        crop_paths : sequence of str or Path
            Paths to cropped object images.
        repeats : int
            Number of question cycles per image.  Questions are iterated
            round-robin across repeats.

        Returns
        -------
        list of str
            De-duplicated candidate prompt strings, in discovery order.
        """
        candidates: set[str] = set()
        ordered: list[str] = []

        for idx, img_path in enumerate(crop_paths):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                logger.warning("Skipping unreadable image: %s", img_path)
                continue

            for cycle in range(repeats):
                question = self.config.questions[cycle % len(self.config.questions)]
                desc = self._ask(image, question)
                if desc and desc not in candidates:
                    candidates.add(desc)
                    ordered.append(desc)

            if (idx + 1) % 5 == 0:
                logger.info(
                    "Processed %d/%d crops (%d candidates so far)",
                    idx + 1,
                    len(crop_paths),
                    len(candidates),
                )

        logger.info("Generated %d unique candidates", len(ordered))
        return ordered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ask(self, image: Image.Image, question: str) -> str | None:
        """Pose *question* about *image* and return the cleaned answer."""
        text = self._PROMPT_TEMPLATE.format(question=question)
        inputs = self._processor(text=text, images=image, return_tensors="pt").to(
            self.device
        )

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

        decoded = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return self._clean(decoded)

    @classmethod
    def _clean(cls, raw: str) -> str | None:
        """Post-process a VLM response into a usable prompt fragment.

        Strips the instruction prefix, removes common filler phrases,
        lowercases, and discards results shorter than 3 characters.
        """
        # Extract the response portion after [/INST]
        if "[/INST]" in raw:
            raw = raw.split("[/INST]")[-1]

        text = raw.strip().lower()

        for filler in cls._FILLER_PHRASES:
            text = text.replace(filler, "")

        text = text.strip(".,!? ")
        return text if len(text) >= 3 else None
