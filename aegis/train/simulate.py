"""Synthetic match generator: humans and cheats, with ground truth.

Why this exists
---------------
An anti-cheat's hardest problem is not the model, it is the labels. Supervised
systems like VACnet depend on human review queues to produce them, which means
the detector is always chasing whatever cheat was popular six months ago, and
a novel technique is invisible until enough of it has been reviewed.

Aegis attacks that in two ways, and both need this module:

  1. **Bootstrap.** A physically-grounded simulator gives exactly-labelled data
     on day zero, including cheat families nobody has reported yet, because we
     can implement the *technique* rather than wait for the sample.
  2. **Red-teaming.** Every new detector is evaluated against an adaptive
     adversary: the humaniser parameters here (smoothing, FOV, jitter,
     injected delay) span the range real cheats expose in their config menus,
     so we can plot detection rate against "how careful is the cheater" and
     know where the detector actually fails.

What is modelled
----------------
The human aim model is the optimised-submovement model from the motor-control
literature (Meyer et al. 1988), which is the reason the features in
``features/aim.py`` work: a ballistic primary movement that deliberately
misses, then visually-guided corrections, all riding on 1/f neuromuscular
noise with a physiological tremor peak.

The cheats are modelled as *mechanisms* -- "read the target's bone position
and interpolate the view toward it over N ticks" -- not as statistical
caricatures. That matters: if the simulator generated cheats by adding
Gaussian noise to human traces, the detector would learn to detect Gaussian
noise, and would fail on contact with reality.

Honest limitations (read before trusting a number produced from this data)
-------------------------------------------------------------------------
* Map geometry is abstract 2D-plus-height, so map-specific priors (common
  angles, off-angles, boost spots) are absent. The wallhack permutation test
  is designed to be robust to those, but its absolute effect sizes here are
  optimistic relative to a real map.
* Real human skill is heavier-tailed than the profile distribution below;
  expect the false-positive rate on real data to be higher than measured here
  until the model is recalibrated on real negatives.
* Network jitter, packet loss, and interpolation artefacts are modelled only
  as view-angle sampling noise. Real telemetry is messier.
* Nobody in this simulation tilts, gets distracted, or plays badly on purpose.

The correct use of this module is to build and rank detectors and to establish
that the pipeline is sound end to end. Operating thresholds must be re-derived
on real negatives before anything is allowed to issue a verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.types import (
    DamageEvent,
    HitGroup,
    KillEvent,
    PlayerFlag,
    PlayerInfo,
    RoundInfo,
    Session,
    ShotEvent,
    Team,
    VisibilityEvent,
    wrap_deg,
)

# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------

EYE = 1.60
HEAD_RADIUS = 0.11  # metres
BODY_RADIUS = 0.34
MAX_SPEED = 4.76  # m/s, CS2 running speed
DT_DEFAULT = 1.0 / 128.0


@dataclass
class Wall:
    """Vertical occluder, a segment in plan view."""

    x1: float
    y1: float
    x2: float
    y2: float


def default_map(rng: np.random.Generator) -> tuple[list[Wall], np.ndarray]:
    """A small arena with cover, plus a set of waypoints.

    Deliberately simple: enough occlusion that visibility transitions are
    frequent and non-trivial, without pretending to be a real map.
    """
    walls = [
        Wall(-16, -16, 16, -16), Wall(16, -16, 16, 16),
        Wall(16, 16, -16, 16), Wall(-16, 16, -16, -16),
        Wall(-6, -6, -6, 4), Wall(6, -4, 6, 6),
        Wall(-2, 8, 8, 8), Wall(-8, -8, 2, -8),
        Wall(-12, 2, -7, 2), Wall(7, -2, 12, -2),
        Wall(0, -12, 0, -5),
    ]
    pts = []
    while len(pts) < 24:
        p = rng.uniform(-14, 14, size=2)
        if all(abs(p[0] - w.x1) > 1.0 or abs(p[1] - w.y1) > 1.0 for w in walls):
            pts.append(p)
    return walls, np.asarray(pts)


def _seg_intersect(p1, p2, q1, q2) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def line_of_sight(a: np.ndarray, b: np.ndarray, walls: list[Wall]) -> bool:
    for w in walls:
        if _seg_intersect(a, b, (w.x1, w.y1), (w.x2, w.y2)):
            return False
    return True


# --------------------------------------------------------------------------
# Player profiles
# --------------------------------------------------------------------------


@dataclass
class AimProfile:
    """Human motor parameters. Ranges chosen to span bronze..pro."""

    skill: float = 0.5  # 0..1, drives everything else
    reaction_mu: float = 0.200
    reaction_sigma: float = 0.045
    reaction_tau: float = 0.080
    # Primary submovement covers this fraction of the error, then corrections.
    primary_gain: float = 0.85
    primary_gain_sd: float = 0.10
    # Peak angular speed of the ballistic phase, deg/s.
    peak_speed: float = 900.0
    # Delay before each corrective submovement, seconds.
    correction_delay: float = 0.075
    correction_gain: float = 0.75
    max_corrections: int = 3
    # Tremor amplitude in deg/s RMS.
    tremor_rms: float = 6.0
    # Fire when angular error is below this (deg), plus impatience.
    fire_threshold: float = 2.2
    # Recoil compensation skill: gain and per-shot noise.
    recoil_gain: float = 0.80
    recoil_gain_sd: float = 0.18
    recoil_lag_s: float = 0.055

    @staticmethod
    def sample(rng: np.random.Generator, skill: float | None = None) -> "AimProfile":
        # Beta(1.8, 1.8) rather than the original Beta(2.2, 2.2): slightly
        # heavier low/high-skill tails so clean players are more heterogeneous
        # and detectors are not tuned against a suspiciously uniform human.
        s = float(rng.beta(1.8, 1.8)) if skill is None else float(skill)
        return AimProfile(
            skill=s,
            reaction_mu=float(np.clip(rng.normal(0.255 - 0.085 * s, 0.022), 0.135, 0.400)),
            reaction_sigma=float(np.clip(rng.normal(0.048 - 0.014 * s, 0.008), 0.015, 0.090)),
            reaction_tau=float(np.clip(rng.normal(0.095 - 0.035 * s, 0.020), 0.025, 0.180)),
            primary_gain=float(np.clip(rng.normal(0.78 + 0.14 * s, 0.05), 0.55, 0.98)),
            primary_gain_sd=float(np.clip(rng.normal(0.14 - 0.06 * s, 0.02), 0.03, 0.25)),
            peak_speed=float(np.clip(rng.normal(650 + 700 * s, 120), 300, 2000)),
            correction_delay=float(np.clip(rng.normal(0.095 - 0.030 * s, 0.015), 0.035, 0.160)),
            correction_gain=float(np.clip(rng.normal(0.62 + 0.22 * s, 0.07), 0.35, 0.95)),
            max_corrections=int(rng.integers(2, 5)),
            tremor_rms=float(np.clip(rng.normal(9.0 - 4.5 * s, 1.8), 1.5, 16.0)),
            fire_threshold=float(np.clip(rng.normal(3.2 - 1.6 * s, 0.5), 0.7, 5.5)),
            recoil_gain=float(np.clip(rng.normal(0.55 + 0.38 * s, 0.10), 0.20, 1.05)),
            recoil_gain_sd=float(np.clip(rng.normal(0.30 - 0.18 * s, 0.05), 0.04, 0.45)),
            recoil_lag_s=float(np.clip(rng.normal(0.075 - 0.030 * s, 0.015), 0.020, 0.140)),
        )


@dataclass
class CheatConfig:
    """A cheat loadout. All-false is a clean player."""

    aimbot: bool = False
    aimbot_fov: float = 30.0  # degrees; only assists inside this cone
    aimbot_smooth_ticks: int = 4  # how many ticks the correction is spread over
    aimbot_strength: float = 1.0  # 0..1 blend toward the solved angle
    aimbot_jitter_deg: float = 0.0  # humaniser: noise added to the solution
    aimbot_delay_s: float = 0.0  # humaniser: artificial reaction delay
    aimbot_retain_tremor: bool = False  # humaniser: keep hand noise during lock
    aimbot_bone_offset: float = 0.0  # aim below the head to look natural

    triggerbot: bool = False
    triggerbot_delay_s: float = 0.020
    triggerbot_delay_jitter: float = 0.005

    wallhack: bool = False
    wallhack_prealign: float = 0.35  # 0..1 how strongly to drift onto unseen foes

    norecoil: bool = False
    norecoil_strength: float = 1.0
    norecoil_jitter: float = 0.0

    bhop: bool = False
    bhop_perfect_rate: float = 0.99

    spinbot: bool = False

    def label(self) -> str:
        on = [k for k in ("aimbot", "triggerbot", "wallhack", "norecoil", "bhop", "spinbot")
              if getattr(self, k)]
        return "+".join(on) if on else "clean"

    @property
    def is_cheating(self) -> bool:
        return bool(self.label() != "clean")


def sample_cheat(rng: np.random.Generator, family: str) -> CheatConfig:
    """Sample a cheat configuration from a named family.

    ``blatant`` and ``legit`` are the two ends of the spectrum that matters:
    a rage config is trivial to catch and a closet config is the actual
    business problem. ``adaptive`` samples the humaniser knobs across their
    full range so evaluation can report detection-vs-stealth curves.
    """
    if family == "clean":
        return CheatConfig()
    if family == "blatant":
        return CheatConfig(aimbot=True, aimbot_fov=180.0, aimbot_smooth_ticks=1,
                           aimbot_strength=1.0, triggerbot=True, wallhack=True,
                           norecoil=True, norecoil_strength=1.0)
    if family == "spinbot":
        return CheatConfig(spinbot=True, aimbot=True, aimbot_fov=180.0,
                           aimbot_smooth_ticks=1, triggerbot=True)
    if family == "aimbot_legit":
        return CheatConfig(
            aimbot=True,
            aimbot_fov=float(rng.uniform(4.0, 14.0)),
            aimbot_smooth_ticks=int(rng.integers(6, 18)),
            aimbot_strength=float(rng.uniform(0.35, 0.75)),
            aimbot_jitter_deg=float(rng.uniform(0.15, 0.6)),
            aimbot_delay_s=float(rng.uniform(0.08, 0.20)),
            aimbot_retain_tremor=True,
            aimbot_bone_offset=float(rng.uniform(0.0, 0.12)),
        )
    if family == "triggerbot":
        return CheatConfig(triggerbot=True,
                           triggerbot_delay_s=float(rng.uniform(0.005, 0.060)),
                           triggerbot_delay_jitter=float(rng.uniform(0.002, 0.015)))
    if family == "wallhack":
        return CheatConfig(wallhack=True, wallhack_prealign=float(rng.uniform(0.20, 0.75)))
    if family == "norecoil":
        return CheatConfig(norecoil=True,
                           norecoil_strength=float(rng.uniform(0.85, 1.0)),
                           norecoil_jitter=float(rng.uniform(0.0, 0.25)))
    if family == "bhop":
        return CheatConfig(bhop=True, bhop_perfect_rate=float(rng.uniform(0.90, 1.0)))
    if family == "adaptive":
        # Uniform over the humaniser space -- the red-team distribution.
        return CheatConfig(
            aimbot=bool(rng.random() < 0.7),
            aimbot_fov=float(rng.uniform(3.0, 40.0)),
            aimbot_smooth_ticks=int(rng.integers(1, 26)),
            aimbot_strength=float(rng.uniform(0.25, 1.0)),
            aimbot_jitter_deg=float(rng.uniform(0.0, 1.0)),
            aimbot_delay_s=float(rng.uniform(0.0, 0.25)),
            aimbot_retain_tremor=bool(rng.random() < 0.7),
            aimbot_bone_offset=float(rng.uniform(0.0, 0.15)),
            triggerbot=bool(rng.random() < 0.35),
            triggerbot_delay_s=float(rng.uniform(0.01, 0.12)),
            wallhack=bool(rng.random() < 0.4),
            wallhack_prealign=float(rng.uniform(0.05, 0.6)),
            norecoil=bool(rng.random() < 0.4),
            norecoil_strength=float(rng.uniform(0.5, 1.0)),
            norecoil_jitter=float(rng.uniform(0.0, 0.5)),
        )
    if family == "aim_smoothing":
        # The hardest humanised profile in the roster: every humaniser knob
        # pushed to the plausible end of its range simultaneously. Where
        # ``aimbot_legit`` samples moderate settings, this is what a
        # "undetectable" marketed config actually looks like -- tiny FOV,
        # heavy smoothing, long artificial delay, tremor retained, aim placed
        # off-bone. Exists so per-family recall is reported against a
        # worst-case closet cheat, not just mid-range ones.
        return CheatConfig(
            aimbot=True,
            aimbot_fov=float(rng.uniform(2.5, 6.0)),
            aimbot_smooth_ticks=int(rng.integers(16, 26)),
            aimbot_strength=float(rng.uniform(0.25, 0.45)),
            aimbot_jitter_deg=float(rng.uniform(0.4, 1.0)),
            aimbot_delay_s=float(rng.uniform(0.18, 0.28)),
            aimbot_retain_tremor=True,
            aimbot_bone_offset=float(rng.uniform(0.10, 0.18)),
            triggerbot=bool(rng.random() < 0.25),
            triggerbot_delay_s=float(rng.uniform(0.04, 0.10)),
        )
    raise ValueError(f"unknown cheat family {family!r}")


CHEAT_FAMILIES = ("clean", "blatant", "spinbot", "aimbot_legit", "triggerbot",
                  "wallhack", "norecoil", "bhop", "adaptive", "aim_smoothing")


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


def pink_noise(n: int, rng: np.random.Generator, beta: float = 1.2) -> np.ndarray:
    """1/f^beta noise via spectral shaping, normalised to unit variance.

    This is what gives simulated humans the DFA exponent and spectral slope
    that ``features/spectral.py`` measures. Using white noise here instead
    would make the spectral channel trivially perfect and the evaluation a
    lie.
    """
    if n < 4:
        return rng.standard_normal(n)
    m = 1 << (int(math.ceil(math.log2(n))) + 1)
    f = np.fft.rfftfreq(m)
    f[0] = f[1]
    spec = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) / np.sqrt(2)
    spec *= f ** (-beta / 2.0)
    spec[0] = 0
    x = np.fft.irfft(spec, n=m)[:n]
    sd = x.std()
    return x / sd if sd > 1e-12 else x


def tremor_signal(n: int, dt: float, rms_deg_s: float, rng: np.random.Generator) -> np.ndarray:
    """Physiological tremor: pink background plus an 8-12 Hz resonance.

    Returned as an angular *velocity* in deg/s; the caller integrates.
    """
    base = pink_noise(n, rng, beta=1.3)
    f0 = float(rng.uniform(8.0, 12.0))
    t = np.arange(n) * dt
    # Amplitude-modulated narrowband component, not a pure tone.
    env = 0.5 + 0.5 * np.abs(pink_noise(n, rng, beta=2.0))
    phase = 2 * np.pi * f0 * t + np.cumsum(rng.standard_normal(n)) * 0.04
    narrow = env * np.sin(phase)
    sig = 0.75 * base + 0.45 * narrow
    sd = sig.std()
    return sig / (sd + 1e-12) * rms_deg_s


def ex_gaussian(rng: np.random.Generator, mu: float, sigma: float, tau: float) -> float:
    return float(rng.normal(mu, sigma) + rng.exponential(tau))


# --------------------------------------------------------------------------
# Recoil
# --------------------------------------------------------------------------

# A deterministic AK-like pattern: strong vertical climb then a horizontal
# S-curve. Values are degrees of accumulated aimpunch per shot index.
def recoil_pattern(n: int) -> np.ndarray:
    i = np.arange(n)
    pitch = -np.minimum(1.9 * np.sqrt(i + 1), 9.0)
    yaw = np.where(i < 6, 0.10 * i, np.where(i < 15, 0.6 - 0.28 * (i - 6), -1.9 + 0.22 * (i - 15)))
    return np.stack([pitch, yaw], axis=1)


RECOIL = recoil_pattern(40)


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


@dataclass
class Agent:
    slot: int
    team: int
    profile: AimProfile
    cheat: CheatConfig
    pos: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    health: int = 100
    alive: bool = True
    respawn_at: float = 0.0
    waypoint: int = 0
    # Aim state machine
    target: int = -1
    target_seen_at: float = float("inf")
    react_until: float = float("inf")
    saccade: list = field(default_factory=list)  # queued (t_start, dur, dp, dy)
    corrections_left: int = 0
    next_correction_at: float = float("inf")
    # Firing
    spray: int = 0
    burst_left: int = 0
    last_shot_t: float = -99.0
    firing_until: float = 0.0
    recoil: np.ndarray = field(default_factory=lambda: np.zeros(2))
    applied_comp: np.ndarray = field(default_factory=lambda: np.zeros(2))
    weapon: int = 1
    on_ground: bool = True
    vz: float = 0.0
    jump_queued_at: float = float("inf")
    kills: int = 0


def _angles_to(src: np.ndarray, dst: np.ndarray, dst_z_off: float = 0.0) -> tuple[float, float]:
    d = dst - src
    yaw = math.degrees(math.atan2(d[1], d[0]))
    horiz = math.hypot(d[0], d[1])
    pitch = -math.degrees(math.atan2(dst_z_off, max(horiz, 1e-6)))
    return pitch, yaw


def _ang_err(p1, y1, p2, y2) -> float:
    dp = p2 - p1
    dy = wrap_deg(y2 - y1) * math.cos(math.radians(np.clip((p1 + p2) / 2, -89, 89)))
    return math.hypot(dp, dy)


# --------------------------------------------------------------------------
# Main simulation
# --------------------------------------------------------------------------


def network_jitter_times(
    t: np.ndarray,
    rng: np.random.Generator,
    max_ms: float = 30.0,
) -> np.ndarray:
    """Bounded random-walk jitter on a monotonic time vector (network delay).

    Models what the behavioural layer actually receives on a real deployment:
    tick timestamps displaced by accumulating network/processing delay, not a
    clean simulation clock. The walk is clamped to +/-``max_ms`` and the
    result is re-monotonised (and clamped >= 0) so downstream timebase
    validation still passes; per-player event ordering within one tick is
    preserved because every row sharing a step gets the same displacement.

    Deliberately NOT applied by default: it exists so evaluation can measure
    detector robustness to realistic transport noise, and so a future ingest
    path can be tested against degraded timestamps.
    """
    t = np.asarray(t, dtype=np.float64)
    if t.size < 2 or max_ms <= 0.0:
        return t.copy()
    sigma_ms = max_ms / 4.0  # walk std; clamping keeps tails inside max_ms
    steps = np.diff(t)
    walk = np.concatenate([
        [0.0],
        np.cumsum(rng.normal(0.0, sigma_ms / 1000.0, size=steps.size)),
    ])
    walk = np.clip(walk, -max_ms / 1000.0, max_ms / 1000.0)
    out = t + walk
    # Re-monotonise: jitter must delay packets, never reorder them.
    return np.maximum.accumulate(np.maximum(out, 0.0))


def simulate_session(
    session_id: str,
    cheat_families: list[str],
    duration_s: float = 180.0,
    dt: float = DT_DEFAULT,
    seed: int = 0,
    game: str = "cs2",
    n_per_team: int | None = None,
    net_jitter_ms: float | None = None,
) -> tuple[Session, dict[int, str]]:
    """Run one synthetic match.

    ``cheat_families`` gives one family name per player; team assignment
    alternates. Returns the canonical ``Session`` and a slot -> label map for
    supervision. ``net_jitter_ms`` (off by default) displaces tick timestamps
    with a bounded random walk before returning -- see
    ``network_jitter_times``.
    """
    rng = np.random.default_rng(seed)
    walls, waypoints = default_map(rng)
    n = len(cheat_families)
    n_steps = int(duration_s / dt)

    agents: list[Agent] = []
    for i, fam in enumerate(cheat_families):
        team = int(Team.SIDE_A) if i % 2 == 0 else int(Team.SIDE_B)
        agents.append(Agent(
            slot=i, team=team,
            profile=AimProfile.sample(rng),
            cheat=sample_cheat(rng, fam),
            pos=waypoints[rng.integers(len(waypoints))].copy(),
            yaw=float(rng.uniform(-180, 180)),
            waypoint=int(rng.integers(len(waypoints))),
        ))

    # Per-agent pre-generated tremor (velocity, deg/s).
    tremor = {a.slot: (tremor_signal(n_steps, dt, a.profile.tremor_rms * 0.6, rng),
                       tremor_signal(n_steps, dt, a.profile.tremor_rms, rng))
              for a in agents}

    # Output buffers.
    tick_arr = np.zeros(n_steps * n, dtype=_tick_dtype())
    ti = 0
    shots: list[ShotEvent] = []
    damages: list[DamageEvent] = []
    kills: list[KillEvent] = []
    visevents: list[VisibilityEvent] = []

    vis_state: dict[tuple[int, int], bool] = {}
    exposure_start: dict[tuple[int, int], float] = {}

    VIS_EVERY = max(1, int(round((1 / dt) / 16)))  # 16 Hz occlusion sampling

    for step in range(n_steps):
        t = step * dt

        # ---- movement -------------------------------------------------
        for a in agents:
            if not a.alive:
                if t >= a.respawn_at:
                    a.alive = True
                    a.health = 100
                    a.pos = waypoints[rng.integers(len(waypoints))].copy()
                    a.spray = 0
                    a.recoil[:] = 0
                    a.target = -1
                continue
            tgt = waypoints[a.waypoint]
            d = tgt - a.pos
            dist = float(np.linalg.norm(d))
            if dist < 1.0:
                a.waypoint = int(rng.integers(len(waypoints)))
            else:
                speed = MAX_SPEED * (0.55 if a.firing_until > t else 1.0)
                a.pos = a.pos + d / dist * speed * dt
            # Jumping / bhop
            if a.cheat.bhop:
                if a.on_ground:
                    # A script jumps on the landing tick with tiny jitter.
                    if rng.random() < a.cheat.bhop_perfect_rate:
                        a.on_ground = False
                        a.vz = 3.4
                elif a.vz <= 0 and rng.random() < 0.35:
                    a.on_ground = True
            else:
                if a.on_ground and rng.random() < 0.004:
                    a.on_ground = False
                    a.vz = 3.4
                elif not a.on_ground:
                    a.vz -= 9.8 * dt
                    if a.vz < -3.4:
                        a.on_ground = True
                        a.vz = 0.0

        # ---- visibility ------------------------------------------------
        if step % VIS_EVERY == 0:
            for a in agents:
                for b in agents:
                    if a.slot == b.slot or a.team == b.team:
                        continue
                    key = (a.slot, b.slot)
                    vis = bool(a.alive and b.alive and line_of_sight(a.pos, b.pos, walls))
                    if vis_state.get(key) != vis:
                        vis_state[key] = vis
                        pt, yw = _angles_to(a.pos, b.pos, 0.0)
                        visevents.append(VisibilityEvent(
                            t=t, tick=step, observer=a.slot, subject=b.slot, visible=vis,
                            angle_error_deg=_ang_err(a.pitch, a.yaw, pt, yw),
                            distance=float(np.linalg.norm(a.pos - b.pos)),
                        ))
                        if vis:
                            exposure_start[key] = t
                        else:
                            exposure_start.pop(key, None)

        # ---- perception + aim ------------------------------------------
        for a in agents:
            if not a.alive:
                _write_tick(tick_arr, ti, t, step, a, dead=True)
                ti += 1
                continue

            # Target selection: normally only visible enemies; a wallhacker
            # also considers occluded ones.
            best, best_err, best_vis = -1, 1e9, False
            for b in agents:
                if b.team == a.team or not b.alive:
                    continue
                vis = vis_state.get((a.slot, b.slot), False)
                if not vis and not (a.cheat.wallhack or a.cheat.aimbot and a.cheat.aimbot_fov >= 180):
                    continue
                pt, yw = _angles_to(a.pos, b.pos, 0.0)
                err = _ang_err(a.pitch, a.yaw, pt, yw)
                if not vis:
                    err += 40.0 * (1.0 - a.cheat.wallhack_prealign)  # wall targets are lower priority
                if err < best_err:
                    best, best_err, best_vis = b.slot, err, vis

            if best != a.target:
                a.target = best
                a.saccade.clear()
                a.corrections_left = 0
                if best >= 0:
                    a.target_seen_at = t
                    delay = ex_gaussian(rng, a.profile.reaction_mu, a.profile.reaction_sigma,
                                        a.profile.reaction_tau)
                    if a.cheat.aimbot:
                        delay = a.cheat.aimbot_delay_s
                    a.react_until = t + delay
                else:
                    a.react_until = float("inf")

            # Base motion: tremor always present unless a full lock suppresses it.
            tp, ty = tremor[a.slot]
            dp_t = tp[step] * dt
            dy_t = ty[step] * dt

            locked = False
            if a.target >= 0 and t >= a.react_until:
                b = agents[a.target]
                pt, yw = _angles_to(a.pos, b.pos, a.cheat.aimbot_bone_offset)

                if a.cheat.aimbot:
                    err = _ang_err(a.pitch, a.yaw, pt, yw)
                    if err <= a.cheat.aimbot_fov:
                        locked = True
                        k = max(1, a.cheat.aimbot_smooth_ticks)
                        w = a.cheat.aimbot_strength / k
                        jp = rng.normal(0, a.cheat.aimbot_jitter_deg)
                        jy = rng.normal(0, a.cheat.aimbot_jitter_deg)
                        a.pitch += (pt + jp - a.pitch) * w
                        a.yaw += wrap_deg(yw + jy - a.yaw) * w
                        if a.cheat.aimbot_retain_tremor:
                            a.pitch += dp_t
                            a.yaw += dy_t

                if not locked:
                    # Human saccade machinery.
                    if not a.saccade and a.corrections_left == 0 and best_err > a.profile.fire_threshold * 0.6:
                        g = float(np.clip(rng.normal(a.profile.primary_gain, a.profile.primary_gain_sd), 0.2, 1.35))
                        dpitch = (pt - a.pitch) * g
                        dyaw = wrap_deg(yw - a.yaw) * g
                        amp = math.hypot(dpitch, dyaw)
                        dur = max(dt * 2, 0.030 + 0.055 * math.log1p(amp) * (1.4 - 0.6 * a.profile.skill))
                        a.saccade.append([t, dur, dpitch, dyaw, 0.0])
                        a.corrections_left = a.profile.max_corrections
                        a.next_correction_at = t + dur + a.profile.correction_delay
                    elif not a.saccade and a.corrections_left > 0 and t >= a.next_correction_at:
                        cur = _ang_err(a.pitch, a.yaw, pt, yw)
                        if cur > 0.35:
                            g = float(np.clip(rng.normal(a.profile.correction_gain, 0.12), 0.15, 1.2))
                            dpitch = (pt - a.pitch) * g
                            dyaw = wrap_deg(yw - a.yaw) * g
                            amp = math.hypot(dpitch, dyaw)
                            dur = max(dt * 2, 0.025 + 0.030 * math.log1p(amp))
                            a.saccade.append([t, dur, dpitch, dyaw, 0.0])
                            a.next_correction_at = t + dur + a.profile.correction_delay
                        a.corrections_left -= 1

            # Advance any in-flight submovement with an asymmetric min-jerk
            # profile (faster acceleration than deceleration).
            if a.saccade:
                st, dur, dpitch, dyaw, done = a.saccade[0]
                u = min(1.0, (t - st) / dur)
                s = _asym_minjerk(u)
                a.pitch += dpitch * (s - done)
                a.yaw += dyaw * (s - done)
                a.saccade[0][4] = s
                if u >= 1.0:
                    a.saccade.pop(0)

            if not locked:
                a.pitch += dp_t
                a.yaw += dy_t

            if a.cheat.spinbot:
                a.yaw += 1800.0 * dt

            # ---- firing ------------------------------------------------
            fired = False
            if a.target >= 0 and t >= a.react_until:
                b = agents[a.target]
                pt, yw = _angles_to(a.pos, b.pos, 0.0)
                err = _ang_err(a.pitch, a.yaw, pt, yw)
                cycle = 0.100
                can_fire = (t - a.last_shot_t) >= cycle and b.alive
                visible = vis_state.get((a.slot, b.slot), False)
                # Continue an in-flight burst on a looser cone: once the
                # trigger is held, the player is spraying and fighting recoil
                # rather than re-acquiring. This is what produces the spray
                # traces the recoil channel needs.
                in_burst = a.burst_left > 0 and (t - a.last_shot_t) < 0.30
                if in_burst:
                    want = visible and err < a.profile.fire_threshold * 3.0
                elif a.cheat.triggerbot:
                    want = err < 1.0 and visible
                elif locked:
                    want = err < max(1.5, a.cheat.aimbot_jitter_deg * 2)
                else:
                    want = err < a.profile.fire_threshold and visible
                if can_fire and want and not in_burst:
                    # Burst length: humans hold for 3-12 rounds; a rage aimbot
                    # taps because it re-solves every shot.
                    if a.cheat.aimbot and a.cheat.aimbot_smooth_ticks <= 2:
                        a.burst_left = int(rng.integers(1, 3))
                    else:
                        a.burst_left = int(np.clip(rng.geometric(0.16), 2, 14))
                if can_fire and want:
                    a.burst_left -= 1
                    fired = True
                    a.last_shot_t = t
                    first = a.spray == 0
                    idx = min(a.spray, len(RECOIL) - 1)
                    a.recoil = RECOIL[idx].copy()
                    shots.append(ShotEvent(
                        t=t, tick=step, slot=a.slot, weapon=a.weapon,
                        pitch=a.pitch, yaw=float(wrap_deg(a.yaw)),
                        recoil_pitch=float(a.recoil[0]), recoil_yaw=float(a.recoil[1]),
                        spread=0.02 * a.spray, is_first_of_spray=first, spray_index=a.spray,
                    ))
                    a.spray += 1
                    a.firing_until = t + 0.15

                    # Resolve the hit against angular hitbox radii.
                    dist = float(np.linalg.norm(a.pos - b.pos))
                    body_ang = math.degrees(math.atan2(BODY_RADIUS, max(dist, 0.5)))
                    head_ang = math.degrees(math.atan2(HEAD_RADIUS, max(dist, 0.5)))
                    bloom = 0.35 * a.spray * (0.0 if a.cheat.norecoil else 1.0)
                    eff = err + abs(rng.normal(0, 0.25 + bloom))
                    if eff < head_ang * 1.15 and rng.random() < 0.5:
                        hg, dmg = int(HitGroup.HEAD), 105.0
                    elif eff < body_ang:
                        hg, dmg = int(HitGroup.CHEST), 27.0
                    elif eff < body_ang * 1.6:
                        hg, dmg = int(HitGroup.STOMACH), 21.0
                    else:
                        hg, dmg = -1, 0.0
                    if hg >= 0:
                        damages.append(DamageEvent(
                            t=t, tick=step, attacker=a.slot, victim=b.slot, damage=dmg,
                            hitgroup=hg, weapon=a.weapon, distance=dist,
                            victim_visible=vis_state.get((a.slot, b.slot), False),
                        ))
                        b.health -= int(dmg)
                        if b.health <= 0:
                            b.alive = False
                            b.respawn_at = t + float(rng.uniform(3.0, 7.0))
                            a.kills += 1
                            key = (a.slot, b.slot)
                            exp_ms = (t - exposure_start[key]) * 1000.0 if key in exposure_start else -1.0
                            kills.append(KillEvent(
                                t=t, tick=step, attacker=a.slot, victim=b.slot, weapon=a.weapon,
                                headshot=hg == int(HitGroup.HEAD), distance=dist,
                                through_smoke=False,
                                through_wall=not vis_state.get(key, False),
                                attacker_flashed=False,
                                victim_exposure_ms=float(exp_ms),
                            ))

            if not fired and (t - a.last_shot_t) > 0.5:
                a.spray = 0
                a.burst_left = 0
                a.recoil[:] = 0

            # ---- recoil compensation -----------------------------------
            if a.spray > 1 and (t - a.last_shot_t) < 0.12:
                idx = min(a.spray - 1, len(RECOIL) - 1)
                prev = RECOIL[max(idx - 1, 0)]
                delta = RECOIL[idx] - prev
                if a.cheat.norecoil:
                    g = a.cheat.norecoil_strength
                    jitter = rng.normal(0, a.cheat.norecoil_jitter * abs(delta).mean() + 1e-9, 2)
                    comp = -delta * g + jitter
                else:
                    g = float(np.clip(rng.normal(a.profile.recoil_gain, a.profile.recoil_gain_sd), 0.0, 1.4))
                    # Human compensation is lagged: apply the previous shot's
                    # delta, not this one's.
                    prev2 = RECOIL[max(idx - 2, 0)]
                    lagged = prev - prev2
                    comp = -lagged * g + rng.normal(0, 0.10, 2)
                a.pitch += float(comp[0])
                a.yaw += float(comp[1])

            a.pitch = float(np.clip(a.pitch, -89.0, 89.0))
            a.yaw = float(wrap_deg(a.yaw))
            _write_tick(tick_arr, ti, t, step, a, dead=False)
            ti += 1

    session = Session(
        session_id=session_id,
        game=game,
        map_name="sim_arena",
        tickrate=1.0 / dt,
        players={
            a.slot: PlayerInfo(slot=a.slot, player_id=f"{session_id}:p{a.slot}",
                               name=f"p{a.slot}", team=a.team,
                               trust_prior=0.5)
            for a in agents
        },
        ticks=tick_arr[:ti],
        shots=shots, damages=damages, kills=kills, visibility=visevents,
        rounds=[RoundInfo(index=0, t_start=0.0, t_end=duration_s, t_freeze_end=0.0, winner=0)],
        meta={"max_run_speed": MAX_SPEED, "synthetic": True, "attested": False},
    )
    labels = {a.slot: a.cheat.label() for a in agents}
    if net_jitter_ms:
        # Jitter the step grid once, then map every tick row through its step
        # index -- rows sharing a step share a timestamp, so per-player
        # ordering is preserved (see network_jitter_times).
        n_steps = int(duration_s / dt)
        grid = network_jitter_times(
            np.arange(n_steps) * dt, rng, max_ms=float(net_jitter_ms)
        )
        session.ticks["t"] = grid[np.clip(session.ticks["tick"], 0, n_steps - 1)]
    session.meta["cheat_configs"] = {a.slot: a.cheat for a in agents}
    session.meta["aim_profiles"] = {a.slot: a.profile for a in agents}
    return session, labels


def _asym_minjerk(u: float) -> float:
    """Displacement fraction at normalised time u, asymmetric min-jerk.

    Symmetric min-jerk is 10u^3 - 15u^4 + 6u^5. Human rapid movements spend
    ~40% of the time accelerating, so we warp time before applying it.
    """
    u = min(max(u, 0.0), 1.0)
    w = u**0.82  # time warp: earlier peak velocity
    return 10 * w**3 - 15 * w**4 + 6 * w**5


def _tick_dtype():
    from ..core.types import TICK_DTYPE

    return TICK_DTYPE


def _write_tick(arr, i, t, step, a: Agent, dead: bool) -> None:
    r = arr[i]
    r["t"] = t
    r["tick"] = step
    r["slot"] = a.slot
    r["px"], r["py"], r["pz"] = a.pos[0], a.pos[1], 0.0
    r["pitch"] = a.pitch
    r["yaw"] = a.yaw
    r["health"] = max(a.health, 0)
    r["weapon"] = a.weapon
    r["aim_target"] = a.target
    flags = PlayerFlag.NONE if dead else PlayerFlag.ALIVE
    if a.on_ground:
        flags |= PlayerFlag.ON_GROUND
    else:
        flags |= PlayerFlag.AIRBORNE
    r["flags"] = int(flags)
    r["buttons"] = (1 << 1) if (not a.on_ground and a.cheat.bhop) else 0


def make_lobby(rng: np.random.Generator, cheat_rate: float = 0.25,
               families: tuple[str, ...] = ("aimbot_legit", "triggerbot", "wallhack",
                                            "norecoil", "adaptive", "blatant",
                                            "aim_smoothing"),
               size: int = 6) -> list[str]:
    """Sample one lobby's worth of family labels."""
    out = []
    for _ in range(size):
        if rng.random() < cheat_rate:
            out.append(str(rng.choice(families)))
        else:
            out.append("clean")
    return out
