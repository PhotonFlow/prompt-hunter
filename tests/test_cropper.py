"""Unit tests for prompt_hunter.cropper."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from prompt_hunter.cropper import InstanceCropper


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal COCO-format dataset for testing.

    Returns (json_path, image_root).
    """
    image_root = tmp_path / "images"
    image_root.mkdir()

    # Create 3 test images (100×100 solid colours)
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        img = np.full((100, 100, 3), color, dtype=np.uint8)
        cv2.imwrite(str(image_root / f"img_{i:03d}.jpg"), img)

    coco = {
        "images": [
            {"id": 1, "file_name": "img_000.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "img_001.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "img_002.jpg", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 1, "name": "vehicle"},
            {"id": 2, "name": "person"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 40, 40]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 90, 90]},
            {"id": 4, "image_id": 3, "category_id": 2, "bbox": [0, 0, 100, 100]},
        ],
    }

    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")
    return json_path, image_root


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestInstanceCropper:
    """Tests for InstanceCropper."""

    def test_load_annotations(
        self, sample_dataset: tuple[Path, Path]
    ) -> None:
        json_path, image_root = sample_dataset
        cropper = InstanceCropper(json_path, image_root)

        assert cropper.num_images == 3
        assert cropper.num_annotations == 4
        assert set(cropper.category_names) == {"vehicle", "person"}

    def test_crop_all_instances(
        self, sample_dataset: tuple[Path, Path], tmp_path: Path
    ) -> None:
        json_path, image_root = sample_dataset
        cropper = InstanceCropper(
            json_path, image_root, output_root=tmp_path / "crops"
        )

        paths = cropper.crop("vehicle")
        assert len(paths) == 2  # 2 vehicle annotations

        # Verify files exist and are readable
        for p in paths:
            assert os.path.isfile(p)
            img = cv2.imread(p)
            assert img is not None

    def test_crop_with_limit(
        self, sample_dataset: tuple[Path, Path], tmp_path: Path
    ) -> None:
        json_path, image_root = sample_dataset
        cropper = InstanceCropper(
            json_path, image_root, output_root=tmp_path / "crops"
        )

        paths = cropper.crop("vehicle", limit=1)
        assert len(paths) == 1

    def test_crop_deterministic_with_seed(
        self, sample_dataset: tuple[Path, Path], tmp_path: Path
    ) -> None:
        json_path, image_root = sample_dataset

        results = []
        for i in range(2):
            out = tmp_path / f"crops_{i}"
            cropper = InstanceCropper(json_path, image_root, output_root=out)
            paths = cropper.crop("vehicle", limit=1, seed=123)
            results.append([os.path.basename(p) for p in paths])

        assert results[0] == results[1], "Same seed should produce same crops"

    def test_crop_nonexistent_class(
        self, sample_dataset: tuple[Path, Path], tmp_path: Path
    ) -> None:
        json_path, image_root = sample_dataset
        cropper = InstanceCropper(
            json_path, image_root, output_root=tmp_path / "crops"
        )

        paths = cropper.crop("nonexistent_class")
        assert paths == []

    def test_crop_boundary_bbox(self, tmp_path: Path) -> None:
        """Bounding boxes that extend beyond image boundaries are clamped."""
        image_root = tmp_path / "images"
        image_root.mkdir()

        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(image_root / "small.jpg"), img)

        coco = {
            "images": [{"id": 1, "file_name": "small.jpg", "width": 50, "height": 50}],
            "categories": [{"id": 1, "name": "obj"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 40, 30, 30]},
            ],
        }
        json_path = tmp_path / "ann.json"
        json_path.write_text(json.dumps(coco))

        cropper = InstanceCropper(
            json_path, image_root, output_root=tmp_path / "crops"
        )
        paths = cropper.crop("obj")
        assert len(paths) == 1

        crop = cv2.imread(paths[0])
        assert crop is not None
        # Crop should be 10x10 (clamped from 40:50, 40:50)
        assert crop.shape[0] == 10
        assert crop.shape[1] == 10


class TestCropperStaticMethods:
    """Tests for InstanceCropper._extract_crop."""

    def test_zero_area_crop_returns_none(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = InstanceCropper._extract_crop(img, [200, 200, 0, 0])
        assert result is None

    def test_valid_crop(self) -> None:
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = InstanceCropper._extract_crop(img, [10, 10, 20, 20])
        assert result is not None
        assert result.shape == (20, 20, 3)
