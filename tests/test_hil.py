"""Unit tests for the HIL snapshot module (no Telegram required)."""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lto_avoid.hil import compare_images, make_diff_image


def _make_solid_png(color: tuple[float, ...], width: int = 50, height: int = 50) -> bytes:
    """Create a solid-color PNG image and return its bytes."""
    arr = np.full((height, width, 4), color, dtype=np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(arr)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=50)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _make_test_png(seed: int = 0, width: int = 50, height: int = 50) -> bytes:
    """Create a deterministic test PNG with random-looking content."""
    rng = np.random.default_rng(seed)
    arr = rng.random((height, width, 3), dtype=np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(arr)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=50)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


class TestSnapshotRoundtrip:
    """Test saving and loading snapshots from disk."""

    def test_snapshot_save_and_load_roundtrip(self, tmp_path: Path):
        """A saved snapshot can be loaded back and compares as identical."""
        image = _make_test_png(seed=42)
        snapshot_path = tmp_path / "test.png"
        snapshot_path.write_bytes(image)

        loaded = snapshot_path.read_bytes()
        assert compare_images(image, loaded) is True


class TestCompareImages:
    """Test exact image comparison."""

    def test_compare_identical_images_match(self):
        """Two identical images should match."""
        image = _make_test_png(seed=1)
        assert compare_images(image, image) is True

    def test_compare_different_images_detect_change(self):
        """Two different images should not match."""
        image_a = _make_test_png(seed=1)
        image_b = _make_test_png(seed=2)
        assert compare_images(image_a, image_b) is False

    def test_single_pixel_change_detected(self):
        """Even a single pixel change should be detected."""
        from PIL import Image

        arr = np.full((50, 50, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        buf_a = io.BytesIO()
        img.save(buf_a, format="png")
        image_a = buf_a.getvalue()

        arr_noisy = arr.copy()
        arr_noisy[10, 10, 0] = 129
        img_noisy = Image.fromarray(arr_noisy)
        buf_b = io.BytesIO()
        img_noisy.save(buf_b, format="png")
        image_b = buf_b.getvalue()

        assert compare_images(image_a, image_b) is False

    def test_compare_different_shapes_not_equal(self):
        """Images with different shapes should not match."""
        from PIL import Image

        img_a = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        buf_a = io.BytesIO()
        img_a.save(buf_a, format="png")

        img_b = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        buf_b = io.BytesIO()
        img_b.save(buf_b, format="png")

        assert compare_images(buf_a.getvalue(), buf_b.getvalue()) is False


class TestDiffImage:
    """Test diff visualization generation."""

    def test_diff_image_highlights_changes(self):
        """make_diff_image should produce a valid PNG."""
        image_a = _make_test_png(seed=1)
        image_b = _make_test_png(seed=2)
        diff_png = make_diff_image(image_a, image_b)

        # Should be valid PNG bytes
        assert diff_png[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(diff_png) > 100

    def test_diff_image_different_shapes(self):
        """make_diff_image handles different-shaped images gracefully."""
        image_a = _make_solid_png((1.0, 0.0, 0.0, 1.0), width=50, height=50)
        image_b = _make_solid_png((0.0, 1.0, 0.0, 1.0), width=100, height=100)
        diff_png = make_diff_image(image_a, image_b)
        assert diff_png[:8] == b"\x89PNG\r\n\x1a\n"
