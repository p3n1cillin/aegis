"""Feature extraction layer.

Importing this package registers every built-in extractor. See
``aegis/features/pipeline.py`` for the entry points:

    build_spec(caps)                    -> FeatureSpec (frozen column order)
    extract_session_features(...)       -> list[SessionFeatures]
    extract_engagement_tensors(...)     -> list[EngagementTensor]
"""

from .base import (  # noqa: F401
    Engagement,
    EngagementExtractor,
    PlayerContext,
    SessionExtractor,
    build_context,
    build_engagements,
)
from .pipeline import (  # noqa: F401
    SEQ_CHANNELS,
    EngagementTensor,
    FeatureSpec,
    SessionFeatures,
    build_spec,
    extract_engagement_tensors,
    extract_session_features,
    stack_features,
)

__all__ = [
    "Engagement",
    "EngagementExtractor",
    "EngagementTensor",
    "FeatureSpec",
    "PlayerContext",
    "SEQ_CHANNELS",
    "SessionExtractor",
    "SessionFeatures",
    "build_context",
    "build_engagements",
    "build_spec",
    "extract_engagement_tensors",
    "extract_session_features",
    "stack_features",
]
