"""Movement and input-timing analysis (bhop / strafe / macro scripts).

Movement cheats are the easiest family to catch and the one most systems get
wrong, because they check for *impossible* movement (perfect bhop chains) and
miss the far more common *scripted* movement, which is possible but has no
human timing jitter.

The discriminative statistic is not "was the jump frame-perfect", it is the
variance of the timing error. A human doing a strong bhop hits the perfect
frame maybe 30% of the time with a wide, skewed miss distribution. A script
hits it 99.9% of the time, and its rare misses are caused by network jitter,
which has a completely different shape than motor error.

Same logic applies to counter-strafing, jump-throw timings, and burst-fire
cadence: a human's inter-event intervals have a coefficient of variation
around 0.10-0.25; a macro's is under 0.02 and often quantised to the frame.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import EXTRACTORS, Capability
from ..core.types import PlayerFlag
from .base import PlayerContext, SessionExtractor, safe_stats, stat_names

BTN_JUMP = 1 << 1
BTN_DUCK = 1 << 2
BTN_ATTACK = 1 << 0
BTN_LEFT = 1 << 9
BTN_RIGHT = 1 << 10


def _rising_edges(mask: np.ndarray) -> np.ndarray:
    if mask.size < 2:
        return np.empty(0, dtype=int)
    return np.where(np.diff(mask.astype(np.int8)) == 1)[0] + 1


def _intervals(t: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if idx.size < 2:
        return np.empty(0)
    return np.diff(t[idx])


def _cv(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3 or abs(x.mean()) < 1e-12:
        return float("nan")
    return float(x.std() / abs(x.mean()))


def _quantisation_fraction(x: np.ndarray, tol: float) -> float:
    """Fraction of values that repeat an exact modal value within ``tol``."""
    x = x[np.isfinite(x)]
    if x.size < 6:
        return float("nan")
    q = np.round(x / tol).astype(np.int64)
    _, counts = np.unique(q, return_counts=True)
    return float(counts.max() / x.size)


@EXTRACTORS.register("movement")
class MovementFeatures(SessionExtractor):
    name = "movement"
    requires = (Capability.POSITIONS, Capability.VELOCITIES, Capability.BUTTONS)
    names = (
        "mv_jump_chain_max",
        "mv_jump_chain_mean",
        "mv_perfect_bhop_rate",
        "mv_land_to_jump_cv",
        "mv_land_to_jump_min_ms",
        "mv_land_to_jump_quant",
        "mv_airspeed_gain_mean",
        "mv_airspeed_gain_max",
        "mv_strafe_sync_mean",
        "mv_strafe_sync_std",
        "mv_strafe_flip_cv",
        "mv_counterstrafe_error_ms",
        "mv_counterstrafe_cv",
        "mv_speed_over_max_frac",
        *stat_names("mv_burst_iti"),
        "mv_burst_iti_cv",
        "mv_button_hold_cv",
        "mv_n_jumps",
    )

    def extract(self, ctx: PlayerContext) -> dict[str, float]:
        out = dict.fromkeys(self.names, float("nan"))
        tk = ctx.ticks
        if tk.size < 64:
            return out

        t = tk["t"].astype(np.float64)
        flags = tk["flags"].astype(np.int64)
        btn = tk["buttons"].astype(np.int64)
        on_ground = (flags & int(PlayerFlag.ON_GROUND)) != 0
        alive = (flags & int(PlayerFlag.ALIVE)) != 0
        speed = np.hypot(tk["vx"].astype(np.float64), tk["vy"].astype(np.float64))

        # --- bunny hopping -------------------------------------------------
        jump_press = _rising_edges((btn & BTN_JUMP) != 0)
        out["mv_n_jumps"] = float(jump_press.size)
        landings = _rising_edges(on_ground)

        if jump_press.size >= 2:
            # A "chain" is consecutive jumps with at most one grounded tick in
            # between -- the definition of a successful bhop link.
            ground_runs = []
            run = 0
            for i in range(1, tk.size):
                if on_ground[i]:
                    run += 1
                elif run:
                    ground_runs.append((i - run, run))
                    run = 0
            chains, cur = [], 0
            for _, length in ground_runs:
                if length <= 2:
                    cur += 1
                else:
                    if cur:
                        chains.append(cur)
                    cur = 0
            if cur:
                chains.append(cur)
            if chains:
                out["mv_jump_chain_max"] = float(max(chains))
                out["mv_jump_chain_mean"] = float(np.mean(chains))
                out["mv_perfect_bhop_rate"] = float(
                    sum(1 for _, l in ground_runs if l <= 1) / max(len(ground_runs), 1)
                )

            # Land-to-jump latency: the timing error we actually care about.
            l2j = []
            for li in landings:
                nxt = jump_press[jump_press > li]
                if nxt.size:
                    dt = t[nxt[0]] - t[li]
                    if dt < 0.5:
                        l2j.append(dt)
            l2j = np.asarray(l2j)
            if l2j.size >= 4:
                out["mv_land_to_jump_cv"] = _cv(l2j)
                out["mv_land_to_jump_min_ms"] = float(l2j.min() * 1000.0)
                out["mv_land_to_jump_quant"] = _quantisation_fraction(l2j, 1.0 / 128.0)

        # --- air strafing --------------------------------------------------
        air = (~on_ground) & alive
        if air.sum() >= 32:
            ds = np.diff(speed)
            gains = ds[air[1:]]
            gains = gains[gains > 0]
            if gains.size:
                out["mv_airspeed_gain_mean"] = float(gains.mean())
                out["mv_airspeed_gain_max"] = float(gains.max())

            # Strafe sync: fraction of airborne ticks where the yaw is turning
            # in the same direction as the strafe key. Humans manage 0.75-0.92;
            # a script is >0.99 with almost no variance.
            yaw_u = np.degrees(np.unwrap(np.radians(tk["yaw"].astype(np.float64))))
            dyaw = np.gradient(yaw_u, t)
            side = tk["move_side"].astype(np.float64)
            m = air & (np.abs(side) > 0.5) & (np.abs(dyaw) > 5.0)
            if m.sum() >= 24:
                agree = np.sign(dyaw[m]) == -np.sign(side[m])
                # Chunk into per-jump windows to get a spread, not just a mean.
                chunks = np.array_split(agree, max(2, agree.size // 32))
                syncs = np.array([c.mean() for c in chunks if c.size >= 8])
                if syncs.size:
                    out["mv_strafe_sync_mean"] = float(syncs.mean())
                    out["mv_strafe_sync_std"] = float(syncs.std())
            flips = _rising_edges(np.sign(side) > 0) if side.size else np.empty(0, dtype=int)
            out["mv_strafe_flip_cv"] = _cv(_intervals(t, flips))

        # --- counter-strafing ----------------------------------------------
        # Time between releasing a direction and the velocity reaching ~0. A
        # human overshoots or undershoots; a script releases on the exact tick.
        left = (btn & BTN_LEFT) != 0
        right = (btn & BTN_RIGHT) != 0
        rel = np.where(np.diff((left | right).astype(np.int8)) == -1)[0] + 1
        errs = []
        for i in rel:
            j = min(i + int(0.35 * 64), tk.size - 1)
            seg = speed[i:j]
            if seg.size < 3:
                continue
            k = int(np.argmin(seg))
            errs.append((t[i + k] - t[i]) * 1000.0)
        if len(errs) >= 5:
            e = np.asarray(errs)
            out["mv_counterstrafe_error_ms"] = float(np.median(e))
            out["mv_counterstrafe_cv"] = _cv(e)

        # Speed above the engine cap (adapter reports max in meta if known).
        vmax = float(ctx.session.meta.get("max_run_speed", 0.0))
        if vmax > 0:
            out["mv_speed_over_max_frac"] = float((speed[alive] > vmax * 1.02).mean())

        # --- trigger cadence -------------------------------------------------
        atk = _rising_edges((btn & BTN_ATTACK) != 0)
        iti = _intervals(t, atk)
        iti = iti[(iti > 0.02) & (iti < 1.5)]
        out.update(safe_stats(iti, "mv_burst_iti"))
        out["mv_burst_iti_cv"] = _cv(iti)

        # Button hold-duration variability across all buttons.
        holds = []
        for bit in (BTN_ATTACK, BTN_JUMP, BTN_DUCK):
            m = (btn & bit) != 0
            up = _rising_edges(m)
            down = np.where(np.diff(m.astype(np.int8)) == -1)[0] + 1
            for u in up:
                d = down[down > u]
                if d.size:
                    holds.append(t[d[0]] - t[u])
        if len(holds) >= 8:
            out["mv_button_hold_cv"] = _cv(np.asarray(holds))
        return out
