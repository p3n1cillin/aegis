"""CS2 weapon table.

Only the properties the detectors need are modelled: whether the weapon is
automatic (drives spray segmentation), its cycle time (drives the "faster than
the weapon can fire" check), and a recoil scale used to normalise compensation
gain across weapons so one model covers all of them.

The authoritative recoil *vector* per shot comes from the server, not from
here -- CS2 recoil is deterministic given the shot index and a per-spray seed,
and the server plugin reports it directly. This table exists so that features
are comparable between an AK and a Vector, whose recoil magnitudes differ by
an order of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Weapon:
    wid: int
    name: str
    automatic: bool
    cycle_time: float  # seconds between shots at full rate
    recoil_scale: float  # relative magnitude, AK-47 = 1.0
    is_sniper: bool = False
    is_pistol: bool = False


# Ids are Aegis-internal and stable; the adapter maps CS2's item definition
# indices onto them. Do not renumber -- ids are baked into trained models via
# the weapon_class sequence channel.
WEAPONS: dict[int, Weapon] = {w.wid: w for w in [
    Weapon(0, "unknown", False, 0.10, 1.0),
    Weapon(1, "ak47", True, 0.100, 1.00),
    Weapon(2, "m4a4", True, 0.090, 0.82),
    Weapon(3, "m4a1_silencer", True, 0.090, 0.72),
    Weapon(4, "awp", False, 1.464, 2.60, is_sniper=True),
    Weapon(5, "ssg08", False, 1.250, 1.40, is_sniper=True),
    Weapon(6, "deagle", False, 0.224, 1.80, is_pistol=True),
    Weapon(7, "usp_silencer", False, 0.170, 0.45, is_pistol=True),
    Weapon(8, "glock", False, 0.150, 0.42, is_pistol=True),
    Weapon(9, "p250", False, 0.150, 0.55, is_pistol=True),
    Weapon(10, "tec9", False, 0.120, 0.70, is_pistol=True),
    Weapon(11, "five_seven", False, 0.150, 0.52, is_pistol=True),
    Weapon(12, "galilar", True, 0.090, 0.78),
    Weapon(13, "famas", True, 0.090, 0.80),
    Weapon(14, "aug", True, 0.090, 0.76),
    Weapon(15, "sg556", True, 0.090, 0.88),
    Weapon(16, "mp9", True, 0.075, 0.40),
    Weapon(17, "mac10", True, 0.075, 0.45),
    Weapon(18, "mp7", True, 0.080, 0.42),
    Weapon(19, "ump45", True, 0.092, 0.48),
    Weapon(20, "p90", True, 0.070, 0.50),
    Weapon(21, "bizon", True, 0.080, 0.38),
    Weapon(22, "nova", False, 0.880, 1.10),
    Weapon(23, "xm1014", False, 0.250, 1.20),
    Weapon(24, "mag7", False, 0.860, 1.15),
    Weapon(25, "negev", True, 0.070, 1.30),
    Weapon(26, "m249", True, 0.080, 1.20),
    Weapon(27, "scar20", False, 0.250, 1.60, is_sniper=True),
    Weapon(28, "g3sg1", False, 0.250, 1.55, is_sniper=True),
    Weapon(29, "revolver", False, 0.400, 1.90, is_pistol=True),
    Weapon(30, "cz75a", False, 0.090, 0.60, is_pistol=True),
    Weapon(31, "dual_berettas", False, 0.120, 0.65, is_pistol=True),
]}

NAME_TO_ID: dict[str, int] = {w.name: w.wid for w in WEAPONS.values()}

# CS2 item definition index -> Aegis weapon id. Extend as Valve adds items;
# anything unmapped falls through to 0 ("unknown"), which is safe because the
# weapon channel is only used for normalisation.
CS2_DEFINDEX_TO_ID: dict[int, int] = {
    7: 1, 16: 2, 60: 3, 9: 4, 40: 5, 1: 6, 61: 7, 4: 8, 36: 9, 30: 10,
    3: 11, 13: 12, 10: 13, 8: 14, 39: 15, 34: 16, 17: 17, 33: 18, 24: 19,
    19: 20, 26: 21, 35: 22, 25: 23, 27: 24, 28: 25, 14: 26, 38: 27, 11: 28,
    64: 29, 63: 30, 2: 31,
}


def weapon_by_name(name: str) -> Weapon:
    n = name.lower().removeprefix("weapon_")
    return WEAPONS.get(NAME_TO_ID.get(n, 0), WEAPONS[0])


def weapon_by_defindex(idx: int) -> Weapon:
    return WEAPONS.get(CS2_DEFINDEX_TO_ID.get(idx, 0), WEAPONS[0])


# Prefer a stable defindex when encoding Session -> plugin frames.
ID_TO_CS2_DEFINDEX: dict[int, int] = {}
for _def, _wid in CS2_DEFINDEX_TO_ID.items():
    ID_TO_CS2_DEFINDEX.setdefault(_wid, _def)


def defindex_for_weapon(wid: int) -> int:
    return int(ID_TO_CS2_DEFINDEX.get(int(wid), 0))
