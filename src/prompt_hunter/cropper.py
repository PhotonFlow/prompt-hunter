"""Instance cropping from COCO-format detection annotations.

Extracts individual object instances from full images using bounding-box
annotations, producing tightly-cropped patches suitable for downstream
visual analysis (prompt mining, feature extraction, etc.).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np


class InstanceCropper:
    """Crops object instances from images using COCO-format annotations.

    Given a COCO JSON annotation file and an image directory, this class
    extracts bounding-box crops for a specified object category.  Crops are
    saved to disk and their paths returned for downstream processing.

    Parameters
    ----------
    annotation_path : str | Path
        Path to a COCO-format JSON file containing ``images``,
        ``annotations``, and ``categories`` fields.
    image_root : str | Path
        Root directory containing the source images referenced in the
        annotation file.
    output_root : str | Path
        Directory under which cropped images will be saved.  Sub-directories
        are created automatically per ``(mode, target_class)`` pair.

    Example
    -------
    >>> cropper = InstanceCropper("annotations.json", "images/", "crops/")
    >>> paths = cropper.crop(target_class="person", limit=50, mode="mining")
    >>> print(f"Cropped {len(paths)} instances")
    """

    def __init__(
        self,
        annotation_path: str | Path,
        image_root: str | Path,
        output_root: str | Path = "cropped_instances",
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.output_root = Path(output_root)

        self._image_map: Dict[int, str] = {}
        self._category_map: Dict[int, str] = {}
        self._annotations: List[Dict[str, Any]] = []
        self._load_annotations()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crop(
        self,
        target_class: str,
        *,
        limit: Optional[int] = None,
        mode: str = "mining",
        seed: int = 42,
    ) -> List[str]:
        """Crop instances of *target_class* and return saved file paths.

        Parameters
        ----------
        target_class : str
            Category name to crop (must match a name in the annotation
            file's ``categories`` array, case-sensitive).
        limit : int or None
            Maximum number of instances to crop.  When *None*, all instances
            are processed in annotation-ID order.  When set, instances are
            randomly sampled (deterministically via *seed*).
        mode : str
            Logical grouping label (e.g. ``"mining"``, ``"evaluation"``).
            Controls the output sub-directory name.
        seed : int
            Random seed used when *limit* triggers sampling.

        Returns
        -------
        list of str
            Absolute paths to the saved crop images.
        """
        save_dir = self.output_root / mode / target_class
        save_dir.mkdir(parents=True, exist_ok=True)

        matching = [
            ann
            for ann in self._annotations
            if self._category_map.get(ann["category_id"]) == target_class
        ]

        if not matching:
            return []

        if limit is None:
            matching.sort(key=lambda a: a["id"])
        else:
            rng = random.Random(seed)
            rng.shuffle(matching)
            matching = matching[:limit]

        saved_paths: List[str] = []
        for ann in matching:
            img_id: int = ann["image_id"]
            file_name = self._image_map.get(img_id)
            if file_name is None:
                continue

            img_path = self.image_root / file_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            crop = self._extract_crop(img, ann["bbox"])
            if crop is None:
                continue

            out_name = f"{ann['id']}_{file_name}"
            out_path = save_dir / out_name
            cv2.imwrite(str(out_path), crop)
            saved_paths.append(str(out_path))

        return saved_paths

    @property
    def num_images(self) -> int:
        """Number of images in the loaded annotation file."""
        return len(self._image_map)

    @property
    def num_annotations(self) -> int:
        """Number of annotations in the loaded annotation file."""
        return len(self._annotations)

    @property
    def category_names(self) -> List[str]:
        """Sorted list of category names in the annotation file."""
        return sorted(set(self._category_map.values()))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_annotations(self) -> None:
        """Parse the COCO JSON file into internal lookup structures."""
        with open(self.annotation_path, "r", encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)

        self._image_map = {img["id"]: img["file_name"] for img in data["images"]}
        self._category_map = {cat["id"]: cat["name"] for cat in data["categories"]}
        self._annotations = data["annotations"]

    @staticmethod
    def _extract_crop(
        image: np.ndarray,
        bbox: Sequence[float],
    ) -> Optional[np.ndarray]:
        """Extract a bounding-box crop, clamped to image boundaries.

        Parameters
        ----------
        image : np.ndarray
            Source image in BGR format (H × W × C).
        bbox : sequence of float
            COCO-format bounding box ``[x, y, width, height]``.

        Returns
        -------
        np.ndarray or None
            Cropped region, or *None* if the resulting crop has zero area.
        """
        x, y, w, h = (int(v) for v in bbox)
        img_h, img_w = image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        crop = image[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
