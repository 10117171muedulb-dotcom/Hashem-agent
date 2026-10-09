"""Experience replay for a design agent.

After every task the agent stores a :class:`Skill`: the prompt, the parameters it
chose, quality metrics and the user's rating.  When a new prompt arrives it
retrieves the most similar past skill and seeds its parameters from it — a simple
but real form of continual improvement that needs no training and no network.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils.log import get_logger
from ..utils.paths import app_data_dir

log = get_logger("learning")

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)

_STOP = {
    "من", "فى", "في", "على", "عن", "الى", "إلى", "ان", "أن", "هو", "هي", "هذا", "هذه",
    "مع", "ثم", "او", "أو", "و", "ل", "ب", "ا", "the", "a", "an", "of", "to", "and", "for",
    "فيديو", "صوره", "صورة", "تصميم", "اعمل", "اصنع", "make", "create", "video", "design",
}


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


@dataclass
class Skill:
    id: str = ""
    kind: str = ""                 # video | image | book | audio
    prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    rating: float = 0.0            # -1 .. +1 from user feedback, 0 = unrated
    times_used: int = 0
    created: float = field(default_factory=time.time)

    def score(self) -> float:
        quality = float(self.metrics.get("quality", 0.5))
        return 0.5 * quality + 0.5 * max(0.0, self.rating)


class SkillStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else app_data_dir() / "skills.json"
        self.skills: list[Skill] = []
        self.load()

    # ------------------------------------------------------------------ persist
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.skills = [Skill(**{k: v for k, v in item.items() if k in Skill.__dataclass_fields__}) for item in raw]
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read skill store (%s)", exc)
            self.skills = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(s) for s in self.skills], ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------ record
    def remember(self, kind: str, prompt: str, params: dict[str, Any], metrics: dict[str, Any] | None = None) -> Skill:
        skill = Skill(id=uuid.uuid4().hex[:10], kind=kind, prompt=prompt, params=params, metrics=metrics or {})
        self.skills.append(skill)
        if len(self.skills) > 500:
            self.skills = self.skills[-500:]
        self.save()
        return skill

    def rate(self, skill_id: str, rating: float) -> Skill | None:
        for skill in self.skills:
            if skill.id == skill_id:
                skill.rating = max(-1.0, min(1.0, float(rating)))
                self.save()
                return skill
        return None

    # ------------------------------------------------------------------ retrieve
    def similar(self, prompt: str, kind: str | None = None, limit: int = 3) -> list[Skill]:
        target = Counter(tokenize(prompt))
        scored = []
        for skill in self.skills:
            if kind and skill.kind != kind:
                continue
            sim = _cosine(target, Counter(tokenize(skill.prompt)))
            if sim > 0.05:
                scored.append((sim * (0.6 + 0.4 * skill.score()), skill))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [skill for _sim, skill in scored[:limit]]

    def best_params(self, prompt: str, kind: str) -> dict[str, Any] | None:
        matches = self.similar(prompt, kind, limit=1)
        if matches and matches[0].score() > 0.4:
            matches[0].times_used += 1
            self.save()
            return dict(matches[0].params)
        return None

    # ----------------------------------------------------------------- insights
    def style_profile(self) -> dict[str, Any]:
        """Aggregate preferences from positively-rated skills."""
        palettes: Counter = Counter()
        styles: Counter = Counter()
        fonts: Counter = Counter()
        for skill in self.skills:
            if skill.rating < 0:
                continue
            params = skill.params or {}
            palette = params.get("palette") or (params.get("colors") or [None])[0]
            if isinstance(palette, str):
                palettes[palette] += 1
            if params.get("style"):
                styles[params["style"]] += 1
            if params.get("font_family"):
                fonts[params["font_family"]] += 1
        return {
            "total_skills": len(self.skills),
            "rated": sum(1 for s in self.skills if s.rating),
            "avg_rating": round(sum(s.rating for s in self.skills) / max(1, len(self.skills)), 2),
            "favourite_palette": palettes.most_common(1)[0][0] if palettes else None,
            "favourite_style": styles.most_common(1)[0][0] if styles else None,
            "top_palettes": dict(palettes.most_common(5)),
        }


def style_profile(store: SkillStore | None = None) -> dict[str, Any]:
    return (store or SkillStore()).style_profile()


__all__ = ["Skill", "SkillStore", "style_profile", "tokenize"]
