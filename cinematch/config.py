"""Configuration déclarative : signaux, poids de base et règles adaptatives.

Les règles sont des données (deltas par signal + seuils), pas du code : on peut
les désactiver une à une (ablation) ou optimiser les poids (Optuna) sans toucher
au moteur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# Ordre canonique des 8 signaux (clés internes, indépendantes de la langue de l'UI)
SIGNALS: tuple[str, ...] = (
    "sbert",
    "keywords",
    "vibe",
    "genre",
    "credits",
    "origin",
    "context",
    "collab",
)
EMBEDDING_SIGNALS: tuple[str, ...] = ("sbert", "vibe")
SPARSE_SIGNALS: tuple[str, ...] = ("keywords", "credits", "genre", "origin", "collab")
RULES: tuple[str, ...] = ("franchise", "horror", "foreign", "popular")

BASE_WEIGHTS: dict[str, float] = {
    "sbert": 0.15,
    "keywords": 0.25,
    "vibe": 0.15,
    "genre": 0.10,
    "credits": 0.10,
    "origin": 0.00,
    "context": 0.05,
    "collab": 0.20,
}


@dataclass(frozen=True)
class ContextConfig:
    """Signal `context` : exp(-|Δ|/σ) sur la popularité et l'année, moyennés."""

    pop_sigma: float = 20.0
    year_sigma: float = 10.0


@dataclass(frozen=True)
class WeightConfig:
    """Poids de base + règles adaptatives au film requête."""

    base: dict[str, float] = field(default_factory=lambda: dict(BASE_WEIGHTS))
    # Animation ou chiffre dans le titre (suites) : sujets et public
    franchise: dict[str, float] = field(default_factory=lambda: {"keywords": 0.15, "collab": 0.10})
    horror: dict[str, float] = field(default_factory=lambda: {"vibe": 0.20, "keywords": 0.10, "credits": -0.05})
    # Langue originale autre que l'anglais
    foreign: dict[str, float] = field(default_factory=lambda: {"origin": 0.30, "sbert": -0.10})
    popular: dict[str, float] = field(default_factory=lambda: {"collab": 0.15, "context": 0.05})
    popular_min_votes: int = 3000
    enabled_rules: frozenset[str] = frozenset(RULES)
    # Cold start : le poids de `collab` est redistribué à parts égales sur ces signaux
    cold_start_targets: tuple[str, ...] = ("sbert", "keywords", "vibe")

    def rule_deltas(self, rule: str) -> dict[str, float]:
        deltas: dict[str, float] = getattr(self, rule)
        return deltas

    def without_rules(self, *rules: str) -> WeightConfig:
        """Copie avec les règles données désactivées (toutes si aucune n'est donnée)."""
        disabled = set(rules) if rules else set(RULES)
        return replace(self, enabled_rules=self.enabled_rules - disabled)

    def with_base(self, base: dict[str, float]) -> WeightConfig:
        return replace(self, base={s: float(base.get(s, 0.0)) for s in SIGNALS})

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base,
            **{rule: self.rule_deltas(rule) for rule in RULES},
            "popular_min_votes": self.popular_min_votes,
            "enabled_rules": sorted(self.enabled_rules),
            "cold_start_targets": list(self.cold_start_targets),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightConfig:
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "enabled_rules" in kwargs:
            kwargs["enabled_rules"] = frozenset(kwargs["enabled_rules"])
        if "cold_start_targets" in kwargs:
            kwargs["cold_start_targets"] = tuple(kwargs["cold_start_targets"])
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: Path) -> WeightConfig:
        return cls.from_dict(json.loads(path.read_text()))
