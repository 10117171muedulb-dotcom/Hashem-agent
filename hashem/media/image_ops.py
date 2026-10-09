"""Image processing built from free, professional primitives (PIL + numpy + optional OpenCV)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from ..utils.log import get_logger

log = get_logger("media.image")

# Type alias kept for a friendlier public API.
ImageResult = Image.Image

_FILTERS = {
    "none", "grayscale", "sepia", "invert", "blur", "sharpen", "warm", "cool",
    "vintage", "noir", "pop", "soft",
}


def filter_names() -> list[str]:
    return sorted(_FILTERS)


def load_image(source: str | Path | bytes) -> Image.Image:
    """Open from a path or raw bytes; always returns RGB/RGBA."""
    if isinstance(source, (bytes, bytearray)):
        image = Image.open(BytesIO(bytes(source)))
    else:
        image = Image.open(source)
    return image.convert("RGBA") if image.mode in {"RGBA", "LA", "PA"} else image.convert("RGB")


def save_image(image: Image.Image, path: str | Path, quality: int = 92) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
    image.save(path, quality=quality)
    return path


def resize(image: Image.Image, width: int | None = None, height: int | None = None, fit: str = "contain") -> Image.Image:
    """Resize with optional aspect control. ``fit`` = contain | stretch | cover."""
    w, h = image.size
    if not width and not height:
        return image
    if fit == "stretch" and width and height:
        return image.resize((width, height), Image.LANCZOS)
    ratio = []
    if width:
        ratio.append(width / w)
    if height:
        ratio.append(height / h)
    scale = max(ratio) if fit == "cover" else min(ratio)
    new = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    if fit == "cover" and width and height:
        left = max(0, (new.width - width) // 2)
        top = max(0, (new.height - height) // 2)
        new = new.crop((left, top, left + width, top + height))
    return new


def crop_smart(image: Image.Image, aspect: float = 1.0) -> Image.Image:
    """Centre crop to a target aspect ratio."""
    w, h = image.size
    target_w, target_h = w, int(w / aspect)
    if target_h > h:
        target_h, target_w = h, int(h * aspect)
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def apply_filter(image: Image.Image, name: str) -> Image.Image:
    """Named photographic filter. Unknown names are returned unchanged."""
    name = (name or "none").lower()
    if name not in _FILTERS or name == "none":
        return image
    rgb = image.convert("RGB")
    if name == "grayscale":
        return rgb.convert("L").convert("RGB")
    if name == "invert":
        return Image.fromarray(255 - np.asarray(rgb), "RGB")
    if name == "blur":
        return rgb.filter(ImageFilter.GaussianBlur(3))
    if name == "sharpen":
        return rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=160))
    if name == "soft":
        return rgb.filter(ImageFilter.GaussianBlur(0.6))
    if name == "sepia":
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        out = np.stack(
            [
                0.393 * r + 0.769 * g + 0.189 * b,
                0.349 * r + 0.686 * g + 0.168 * b,
                0.272 * r + 0.534 * g + 0.131 * b,
            ],
            axis=-1,
        )
        return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), "RGB")
    if name == "noir":
        mono = rgb.convert("L")
        mono = ImageEnhance.Contrast(mono).enhance(1.4)
        return mono.convert("RGB")
    if name == "warm":
        arr = np.asarray(rgb, dtype=np.float32)
        arr[..., 0] = np.clip(arr[..., 0] * 1.12, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.9, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGB")
    if name == "cool":
        arr = np.asarray(rgb, dtype=np.float32)
        arr[..., 2] = np.clip(arr[..., 2] * 1.12, 0, 255)
        arr[..., 0] = np.clip(arr[..., 0] * 0.92, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGB")
    if name == "vintage":
        return ImageEnhance.Contrast(apply_filter(rgb, "sepia")).enhance(0.9)
    if name == "pop":
        rgb = ImageEnhance.Color(rgb).enhance(1.35)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.15)
        return rgb
    return rgb


def auto_enhance(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Balanced one-tap enhancement: exposure, colour, contrast and sharpness."""
    rgb = image.convert("RGB")
    strength = max(0.0, min(2.0, strength))
    rgb = ImageEnhance.Color(rgb).enhance(1.0 + 0.25 * strength)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.0 + 0.15 * strength)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.0 + 0.6 * strength)
    return rgb


def add_border(image: Image.Image, color: tuple | str = "#ffffff", width: int = 24) -> Image.Image:
    from ..motion.colors import parse_color

    rgba = parse_color(color)
    bordered = Image.new(image.mode, (image.width + width * 2, image.height + width * 2), rgba)
    bordered.paste(image, (width, width))
    return bordered


def rounded_corners(image: Image.Image, radius: int = 48) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, image.width, image.height], radius=radius, fill=255)
    out = image.convert("RGBA")
    out.putalpha(mask)
    return out


def overlay_watermark(image: Image.Image, text: str, opacity: float = 0.35, size_ratio: float = 0.05) -> Image.Image:
    """Subtle tiled watermark across the image."""
    from ..motion.fonts import resolve_font
    from ..motion.text import load_font

    base = image.convert("RGBA")
    size = max(12, int(min(image.size) * size_ratio))
    font = load_font(resolve_font(text), size)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    step_x, step_y = size * 6, size * 6
    for y in range(-size, base.height, step_y):
        for x in range(-size, base.width, step_x):
            draw.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * opacity)))
    return Image.alpha_composite(base, overlay)


def remove_background(image: Image.Image, tolerance: float = 0.22, feather: int = 2) -> Image.Image:
    """Chroma-key style background removal.

    Estimates the background colour from the corners and builds an alpha mask from
    per-pixel colour distance.  Works well on product/flat backgrounds — no model
    needed, fully local and free.
    """
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)[..., :3]
    h, w, _ = arr.shape
    corners = [arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]]
    bg = np.mean(np.stack(corners), axis=0)
    distance = np.linalg.norm(arr - bg, axis=-1) / np.sqrt(3 * 255**2)
    alpha = np.clip((distance - tolerance) / max(0.02, tolerance), 0.0, 1.0)
    if feather > 0:
        from PIL import ImageFilter as _F

        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), "L").filter(_F.GaussianBlur(feather))
        alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
    rgba_arr = np.asarray(rgba, dtype=np.float32)
    rgba_arr[..., 3] *= alpha
    return Image.fromarray(np.clip(rgba_arr, 0, 255).astype(np.uint8), "RGBA")


def embed_qr(image: Image.Image, data: str, box_ratio: float = 0.25, position: str = "bottom-right", margin: int = 24) -> Image.Image:
    """Stamp a QR code onto a corner of the image."""
    try:
        import qrcode
    except Exception as exc:  # noqa: BLE001
        log.warning("qrcode unavailable (%s)", exc)
        return image
    qr = qrcode.make(data)
    size = max(64, int(min(image.size) * box_ratio))
    qr_img = qr.convert("RGBA").resize((size, size), Image.NEAREST)
    base = image.convert("RGBA")
    pad = Image.new("RGBA", (size + margin, size + margin), (255, 255, 255, 235))
    pad.alpha_composite(qr_img, (margin // 2, margin // 2))
    x = base.width - pad.width - margin if "right" in position else margin
    y = base.height - pad.height - margin if "bottom" in position else margin
    base.alpha_composite(pad, (x, y))
    return base


def blend_two(background: Image.Image, foreground: Image.Image, mode: str = "center") -> Image.Image:
    """Composite foreground over background (used for cut-outs on posters)."""
    base = background.convert("RGBA")
    fg = foreground.convert("RGBA")
    base.alpha_composite(fg, ((base.width - fg.width) // 2, (base.height - fg.height) // 2))
    return base


__all__ = [
    "ImageResult",
    "filter_names",
    "load_image",
    "save_image",
    "resize",
    "crop_smart",
    "apply_filter",
    "auto_enhance",
    "add_border",
    "rounded_corners",
    "overlay_watermark",
    "remove_background",
    "embed_qr",
    "blend_two",
]
