"""Game adapter contract.

An adapter's only job is to turn a game's native telemetry into
``aegis.core.types.Session`` and to declare honestly what it can provide.
Everything above this line -- features, models, fusion, policy -- is
game-agnostic, which is what makes adding a title a days-long job instead of a
rewrite.

Rules an adapter must follow (violations produce silently wrong detections,
so ``validate_session`` enforces the checkable ones):

  1. Units are metres, seconds, degrees. Convert in the adapter, never later.
  2. Yaw is wrapped to [-180, 180) and pitch to [-90, 90], with negative pitch
     meaning "looking up". CS2's native convention already matches; other
     engines do not.
  3. Time starts at 0 at session start and is monotonic per player.
  4. ``aim_target`` is -1 when unknown. Never 0, which is a valid slot.
  5. If the engine cannot supply a signal, do not synthesise it -- omit the
     capability instead. A zero-filled channel is worse than a missing one,
     because the model will learn the zeros.
"""

from __future__ import annotations

import abc
from typing import Any, Iterable, Iterator

import numpy as np

from ..core.registry import Capability
from ..core.types import TICK_DTYPE, PlayerFlag, Session


class GameAdapter(abc.ABC):
    """Base class for all game adapters."""

    key: str = "unknown"
    display_name: str = "Unknown"
    #: Capabilities this adapter can provide. Gates which extractors run.
    capabilities: frozenset[Capability] = frozenset()
    #: Native units -> metres. CS2 uses Hammer units (1 unit ~= 0.01905 m).
    unit_scale: float = 1.0
    #: Engine max ground speed in m/s, used by movement features.
    max_run_speed: float = 0.0

    @abc.abstractmethod
    def parse(self, source: Any) -> Session:
        """Lower one native source (demo file, replay, live feed) to a Session."""

    def stream(self, source: Any) -> Iterator[Session]:
        """Optional: yield sessions incrementally from a continuous feed."""
        yield self.parse(source)

    # -- helpers available to subclasses -----------------------------------

    def new_session(self, session_id: str, **kw) -> Session:
        s = Session(session_id=session_id, game=self.key, **kw)
        s.meta["max_run_speed"] = self.max_run_speed
        s.meta["unit_scale"] = self.unit_scale
        s.meta["capabilities"] = sorted(c.value for c in self.capabilities)
        return s

    @staticmethod
    def pack_ticks(rows: Iterable[dict]) -> np.ndarray:
        """Build the columnar tick array from dict rows.

        Missing fields take dtype-appropriate defaults; ``aim_target`` defaults
        to -1 rather than 0 (see rule 4).
        """
        rows = list(rows)
        arr = np.zeros(len(rows), dtype=TICK_DTYPE)
        arr["aim_target"] = -1
        for i, r in enumerate(rows):
            for k, v in r.items():
                if k in TICK_DTYPE.names:
                    arr[i][k] = v
        return arr


class AdapterValidationError(ValueError):
    pass


_HIGH_RATE_MIN_HZ = 60.0  # matches the >= 64 Hz contract in Capability docs, minus jitter slack


def _has_high_rate_view(session: Session) -> bool:
    """Measure the actual view-sample density instead of trusting metadata."""
    ts = np.unique(session.ticks["t"]) if session.ticks.size else np.empty(0)
    if ts.size < 2:
        return False
    dt = np.median(np.diff(ts))
    return dt > 0 and (1.0 / dt) >= _HIGH_RATE_MIN_HZ


def effective_capabilities(adapter: GameAdapter, session: Session) -> set[Capability]:
    """Capabilities actually usable for this session, not the adapter's default.

    A capture-time gap (no TraceShape on this map, an unattested client, a
    seq-gap window) makes a capability unusable for one session without the
    adapter losing it for every session. Callers that skip this and check
    ``adapter.capabilities`` directly will run extractors against a channel
    that is honestly empty for this session -- see the module docstring's
    rule 5. Kept identical to the narrowing in ``service/__init__.py`` and
    ``service/ingest.py`` on purpose; change all three together.
    """
    caps = set(adapter.capabilities)
    if session.meta.get("no_vis") or not session.visibility:
        caps.discard(Capability.VISIBILITY)
    if not session.meta.get("attested", False):
        caps.discard(Capability.HID_INPUT)
    if Capability.HIGH_RATE_VIEW in caps and not _has_high_rate_view(session):
        # A title (or a build of one) can declare 64 Hz and actually ship
        # less -- CS2's plugin throttled to 16 Hz for a long time and every
        # session still claimed HIGH_RATE_VIEW. Measure the real sample
        # density rather than trust the declared rate, or the spectral
        # tremor channel silently degrades to noise on real matches while
        # looking fine on full-rate synthetic data.
        caps.discard(Capability.HIGH_RATE_VIEW)
    if session.meta.get("partial"):
        caps.discard(Capability.HID_INPUT)
    return caps


def validate_session(session: Session, adapter: GameAdapter, strict: bool = True) -> list[str]:
    """Check adapter output against the schema contract.

    Returns a list of problems. With ``strict`` it raises instead. Run this in
    CI for every adapter and on a sample of live sessions -- a units bug in an
    adapter is indistinguishable from cheating to every layer above it.
    """
    problems: list[str] = []
    t = session.ticks
    if t.size == 0:
        problems.append("session has no ticks")
    else:
        if np.any(~np.isfinite(t["pitch"])) or np.any(~np.isfinite(t["yaw"])):
            problems.append("non-finite view angles")
        if np.nanmax(np.abs(t["pitch"])) > 90.5:
            problems.append(f"pitch out of range: max |pitch| = {np.nanmax(np.abs(t['pitch'])):.1f}")
        if np.nanmax(np.abs(t["yaw"])) > 180.5:
            problems.append(f"yaw not wrapped: max |yaw| = {np.nanmax(np.abs(t['yaw'])):.1f}")
        if t["t"].min() < -1e-6:
            problems.append("negative timestamps")
        for slot in session.slots():
            st = session.ticks_for(slot)
            if st.size > 1 and np.any(np.diff(st["t"]) < 0):
                problems.append(f"non-monotonic time for slot {slot}")
        speeds = np.hypot(t["vx"], t["vy"])
        if adapter.max_run_speed > 0 and np.nanmedian(speeds[speeds > 0.1]) > adapter.max_run_speed * 3:
            problems.append("velocities look like they are in native units, not m/s")
        if np.any((t["flags"] & int(PlayerFlag.ALIVE)) != 0) is np.False_:
            problems.append("no player is ever flagged ALIVE")

    known = set(session.players)
    for k in session.kills:
        if k.attacker not in known or k.victim not in known:
            problems.append(f"kill references unknown slot {k.attacker}->{k.victim}")
            break
    caps = effective_capabilities(adapter, session)
    if Capability.VISIBILITY in caps and not session.visibility:
        problems.append("adapter claims VISIBILITY but emitted no visibility events")
    if Capability.RECOIL_STATE in caps and session.shots:
        if all(s.recoil_pitch == 0.0 and s.recoil_yaw == 0.0 for s in session.shots):
            problems.append("adapter claims RECOIL_STATE but all recoil is zero")

    if problems and strict:
        raise AdapterValidationError("; ".join(problems))
    return problems
