"""CineMatch : recommandeur de films hybride et explicable."""

from cinematch.config import SIGNALS, WeightConfig
from cinematch.data import Artifacts, load_artifacts
from cinematch.engine import Recommendation, RecommendationResult, recommend, smart_weights

__all__ = [
    "SIGNALS",
    "Artifacts",
    "Recommendation",
    "RecommendationResult",
    "WeightConfig",
    "load_artifacts",
    "recommend",
    "smart_weights",
]
