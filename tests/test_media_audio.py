"""Tests for media processing and procedural audio."""

from __future__ import annotations

import numpy as np
from PIL import Image

from hashem.media import image_ops
from hashem import audio


def _gradient_image(size=(64, 64)):
    return Image.fromarray(
        np.stack([np.linspace(0, 255, size[0], dtype=np.uint8)] * size[1]).T[..., None].repeat(3, axis=2), "RGB"
    )


def test_filters_and_resize():
    image = _gradient_image()
    for name in image_ops.filter_names():
        out = image_ops.apply_filter(image, name)
        assert out.size == image.size
    resized = image_ops.resize(image, width=32)
    assert resized.width == 32


def test_remove_background_flat():
    arr = np.full((40, 40, 3), 200, dtype=np.uint8)
    arr[15:25, 15:25] = [255, 0, 0]  # red square on grey
    image = Image.fromarray(arr, "RGB")
    cut = image_ops.remove_background(image, tolerance=0.2)
    alpha = np.asarray(cut)[..., 3]
    # centre (red) should stay opaque, corners (grey) should be transparent
    assert alpha[20, 20] > 200
    assert alpha[2, 2] < 60


def test_watermark_and_qr():
    image = _gradient_image((120, 120))
    wm = image_ops.overlay_watermark(image, "هاشم")
    assert wm.size == image.size
    qr = image_ops.embed_qr(image, "https://example.com")
    assert qr.size == image.size


def test_music_render_and_wav(tmp_path):
    for style in ["corporate", "upbeat", "lofi", "ambient"]:
        samples = audio.render_music(2.0, style=style)
        assert samples.ndim == 2 and samples.shape[1] == 2
        assert float(np.abs(samples).max()) <= 1.0
    path = audio.save_wav(tmp_path / "m.wav", audio.render_music(1.0))
    assert path.stat().st_size > 1000
    loaded, rate = audio.load_wav(path)
    assert rate == audio.SAMPLE_RATE


def test_mix_with_ducking():
    music = audio.render_music(2.0, style="ambient")
    voice = np.zeros((audio.SAMPLE_RATE, 2), dtype=np.float32)
    voice[:, 0] = 0.5
    mixed = audio.mix_tracks(voice, music, music_volume=0.3)
    assert mixed.shape[0] >= music.shape[0]
