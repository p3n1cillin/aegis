"""Detection models.

Three complementary scorers feed the fusion layer:

  * ``TabularGBDT``      session-level features -> P(cheat)
  * ``TemporalEncoder``  engagement sequences  -> P(cheat | engagement)
  * ``NoveltyAE``        reconstruction error  -> novelty score (open-set)

``CalibratedEnsemble`` wraps them, produces calibrated probabilities and
log-likelihood ratios, and is the only thing the fusion / policy layers talk
to. Load a saved ensemble with ``CalibratedEnsemble.load(path)``.
"""

from .calibrate import ConformalGate, ProbabilityCalibrator
from .ensemble import CalibratedEnsemble, ModelBundle, Score
from .novelty import NoveltyAE
from .tabular import TabularGBDT
from .temporal import TemporalEncoder

__all__ = [
    "CalibratedEnsemble",
    "ConformalGate",
    "ModelBundle",
    "NoveltyAE",
    "ProbabilityCalibrator",
    "Score",
    "TabularGBDT",
    "TemporalEncoder",
]
