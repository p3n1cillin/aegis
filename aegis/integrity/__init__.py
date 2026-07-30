"""Client integrity and HID-level analysis.

Behavioural detection catches aim/wall/trigger assistance that leaves a
statistical footprint in telemetry. It cannot catch a cheat that never aims
for the player -- a radar hack used only for information, or a fully external
tool that the behavioural channels have not yet characterised.

The integrity layer is the complementary half:

  * ``Attestor`` issues unpredictable challenges the client module must answer
    with a signed measurement of its own memory / module list / input path.
  * ``HIDAnalyzer`` inspects raw mouse reports (when attested) for the timing
    signatures of injected input -- synthetic devices have fundamentally
    different inter-report distributions than a physical sensor.

Failed attestation does not equal a ban by itself (laptops drop packets; VPN
users have latency). It *does* mark the session untrusted, which zeroes every
client-sourced feature and raises the behavioural prior. Confirmed tamper
(forged response, impossible measurement) is a deterministic integrity ban.
"""

from .attest import AttestationResult, Attestor, Challenge
from .hid import HIDAnalyzer, HIDReport

__all__ = [
    "AttestationResult",
    "Attestor",
    "Challenge",
    "HIDAnalyzer",
    "HIDReport",
]
