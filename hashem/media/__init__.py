"""Media ingestion and processing: images, video and text."""

from __future__ import annotations

from .image_ops import (
    ImageResult,
    add_border,
    apply_filter,
    auto_enhance,
    crop_smart,
    embed_qr,
    load_image,
    overlay_watermark,
    remove_background,
    resize,
    rounded_corners,
    save_image,
)
from .ingest import (
    detect_kind,
    ingest,
    read_text_file,
    split_paragraphs,
    text_from_any,
)
from .video_ops import (
    concat,
    extract_frames,
    probe,
    thumbnail,
    trim,
    video_to_gif,
)

__all__ = [
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
    "ImageResult",
    "probe",
    "trim",
    "concat",
    "thumbnail",
    "extract_frames",
    "video_to_gif",
    "ingest",
    "detect_kind",
    "read_text_file",
    "text_from_any",
    "split_paragraphs",
]
