"""Central configuration.

Deliberately plain dataclasses rather than a config framework: every knob here
changes detector behaviour, so they need to be greppable and diffable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FeatureConfig:
    # Resampling grid for view-angle signals. Server tick is 64/128 and varies
    # between titles; we resample onto a fixed grid so one model works across
    # games and tickrates.
    resample_hz: float = 128.0
    # Window (seconds) around each engagement used for kinematic features.
    engagement_pre_s: float = 1.0
    engagement_post_s: float = 0.35
    # Minimum samples before a window is considered usable.
    min_window_samples: int = 24
    # Savitzky-Golay smoothing for derivative estimation.
    sg_window: int = 9
    sg_poly: int = 3
    # Spectral analysis band of interest (Hz). Human tremor lives at 8-12 Hz.
    spectral_band: tuple[float, float] = (2.0, 40.0)
    # Snap detection threshold: angular speed (deg/s) above which motion is a
    # candidate "snap" rather than tracking.
    snap_speed_deg_s: float = 400.0


@dataclass
class ModelConfig:
    # Temporal encoder
    seq_len: int = 128
    seq_channels: int = 14
    hidden: int = 96
    n_blocks: int = 4
    n_heads: int = 4
    dropout: float = 0.15
    # Autoencoder latent size for novelty detection
    latent: int = 24
    # Training
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 30
    device: str = "cpu"
    seed: int = 1337


@dataclass
class FusionConfig:
    """Sequential probability ratio test parameters.

    We accumulate log-likelihood-ratio evidence across engagements, rounds and
    sessions rather than making a per-snapshot call. alpha is the *system-wide*
    false-positive budget; a ban is only issued once accumulated evidence
    crosses the corresponding boundary.
    """

    alpha: float = 1e-5  # P(ban | innocent) target -- 1 in 100k
    beta: float = 0.15  # P(miss | cheater) target at the boundary
    # Per-evidence-unit LLR clamp. Prevents one freak engagement from
    # dominating; real cheating produces many mildly-positive units, not one
    # enormous one.
    llr_clip: float = 2.5
    # Evidence half-life in days; behaviour from six months ago should not
    # convict someone today.
    half_life_days: float = 45.0
    # Minimum independent evidence units before any verdict above "watch".
    min_units_flag: int = 12
    min_units_ban: int = 60
    # Correlation discount: evidence units from the same match are not
    # independent. Effective count = n ** corr_exponent within a session.
    corr_exponent: float = 0.65


@dataclass
class PolicyConfig:
    # Verdict bands on posterior probability of cheating.
    watch_threshold: float = 0.60
    review_threshold: float = 0.90
    ban_threshold: float = 0.995
    # Never auto-ban below this many observed engagements, regardless of score.
    min_engagements_for_ban: int = 40
    # Integrity violations are handled separately from behaviour: a confirmed
    # tamper detection is deterministic, not probabilistic.
    integrity_ban_on_confirmed: bool = True
    # Shadow mode: compute everything, issue nothing. Default ON so a fresh
    # deployment cannot ban anyone until an operator explicitly opts in.
    shadow_mode: bool = True


@dataclass
class IntegrityConfig:
    challenge_interval_s: float = 30.0
    challenge_jitter_s: float = 12.0
    max_response_latency_ms: float = 1500.0
    nonce_bytes: int = 32
    # Number of consecutive missed/invalid challenges before the session is
    # marked untrusted (and behavioural HID features are discarded).
    untrusted_after_failures: int = 3


@dataclass
class AegisConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    model_dir: str = "models"

    @classmethod
    def load(cls, path: str | Path) -> "AegisConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            features=FeatureConfig(**raw.get("features", {})),
            model=ModelConfig(**raw.get("model", {})),
            fusion=FusionConfig(**raw.get("fusion", {})),
            policy=PolicyConfig(**raw.get("policy", {})),
            integrity=IntegrityConfig(**raw.get("integrity", {})),
            model_dir=raw.get("model_dir", "models"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


DEFAULT = AegisConfig()
