"""Generic-path presets for titles Aegis has no live access to.

CS2 graduated from the generic path to a first-class adapter (`cs2/`) only
after a real MetaMod:Source2 plugin gave it live telemetry to verify against
-- see `plugins/cs2/SDK_WIRING.md` for how many of that adapter's assumptions
turned out to be wrong until checked against a running server.

Call of Duty (Warzone), Escape from Tarkov, Apex Legends and Battlefield have
no equivalent: no public server-side plugin API, no dedicated-server binary
Aegis can attach to, no SDK. There is no legitimate way to pull live telemetry
from them, and reverse-engineering a client to get it would be exactly the
kind of unauthorised access into a live anti-cheat-protected service this
project should not do. These presets are the same on-ramp CS2 started from:
they let a studio integration begin with a config change instead of a code
change. Every unit_scale / max_run_speed / tick_hz below is a placeholder
labelled as such, not a verified engine constant -- do not cite these as
"Aegis supports Warzone" claims. What *is* verified is the claim underneath
them: the same feature/model/fusion stack that scores a CS2 session scores
these without modification (see `tests/test_invariants.py
::test_generic_presets_score_without_retraining`).
"""

from __future__ import annotations

from ..core.registry import ADAPTERS
from .generic_fps import GenericFPSAdapter

# Respawn's Source-derived engine is architecturally the closest relative in
# this list -- same lineage as CS2's Source2, so the CS2 adapter's assumptions
# (Hammer-unit-style scale, similar movement tuning) are a more defensible
# starting point than a blind default. Apex's server tick rate has been a
# long-running community discussion point without an authoritative public
# number Aegis can cite; treat any tick_hz used against this preset as a
# placeholder to be replaced by whatever an integration actually measures.
apex_legends = GenericFPSAdapter(game_key="apex_legends", unit_scale=0.01905, max_run_speed=6.5)

# IW engine (Call of Duty). Warzone's movement speed and TTK are both faster
# than CS2's; unit_scale=1.0 assumes the integration layer hands Aegis
# pre-converted metres rather than asserting a specific native scale we have
# no public source for.
warzone = GenericFPSAdapter(game_key="warzone", unit_scale=1.0, max_run_speed=7.5)

# Unity, meters-native by convention. Tarkov's combat is slower and more
# positional than an arena shooter's; max_run_speed reflects sprint, not the
# ADS-and-hold playstyle the spectral/reaction channels would actually see
# most engagements in.
tarkov = GenericFPSAdapter(game_key="tarkov", unit_scale=1.0, max_run_speed=6.0)

# Frostbite (DICE/EA), meters-native by convention. Infantry combat only --
# vehicles are a distinct feature domain this stack does not model yet.
battlefield = GenericFPSAdapter(game_key="battlefield", unit_scale=1.0, max_run_speed=7.0)

for _preset in (apex_legends, warzone, tarkov, battlefield):
    ADAPTERS.register(_preset.key)(_preset)

__all__ = ["apex_legends", "warzone", "tarkov", "battlefield"]
