"""Challenge-response attestation for the Aegis client module.

Protocol (summarised; the on-wire schema lives with the client SDK):

  1. Server emits ``Challenge(nonce, deadline, probes)`` on a jittered interval.
  2. Client measures the requested probes (module hashes, input-stack
     integrity bits, a HMAC over the nonce with a device-bound key) and
     returns a signed response before the deadline.
  3. Server verifies signature, freshness, and probe consistency against the
     expected baseline for that client build.

This module implements the *server* side. The client is a separate binary; we
only need the verification logic and the state machine that decides when a
session becomes untrusted.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field

from ..core.config import IntegrityConfig


@dataclass(slots=True)
class Challenge:
    challenge_id: str
    nonce: bytes
    issued_at: float
    deadline: float
    probes: tuple[str, ...] = ("module_hash", "input_stack", "hmac")


@dataclass
class AttestationResult:
    ok: bool
    reason: str
    latency_ms: float = 0.0
    detail: dict = field(default_factory=dict)


@dataclass
class SessionIntegrity:
    player_id: str
    session_id: str
    attested: bool = False
    failures: int = 0
    successes: int = 0
    untrusted: bool = False
    confirmed_tamper: bool = False
    last_challenge: Challenge | None = None
    history: list[AttestationResult] = field(default_factory=list)


class Attestor:
    def __init__(self, cfg: IntegrityConfig | None = None, server_key: bytes | None = None):
        self.cfg = cfg or IntegrityConfig()
        # In production this is an HSM-backed key. Here it is ephemeral unless
        # the operator passes one in -- fine for synthetic / lab use.
        self.server_key = server_key or hashlib.sha256(b"aegis-dev-key").digest()
        self._sessions: dict[str, SessionIntegrity] = {}

    def state(self, session_id: str, player_id: str = "") -> SessionIntegrity:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionIntegrity(
                player_id=player_id, session_id=session_id
            )
        return self._sessions[session_id]

    def issue(self, session_id: str, player_id: str = "", now: float | None = None) -> Challenge:
        now = time.time() if now is None else now
        jitter = secrets.randbelow(max(1, int(self.cfg.challenge_jitter_s * 1000))) / 1000.0
        ch = Challenge(
            challenge_id=secrets.token_hex(8),
            nonce=secrets.token_bytes(self.cfg.nonce_bytes),
            issued_at=now,
            deadline=now + self.cfg.max_response_latency_ms / 1000.0 + jitter * 0.01,
        )
        st = self.state(session_id, player_id)
        st.last_challenge = ch
        return ch

    def expected_mac(self, nonce: bytes, client_measurement: bytes) -> bytes:
        return hmac.new(self.server_key, nonce + client_measurement, hashlib.sha256).digest()

    def verify(
        self,
        session_id: str,
        *,
        challenge_id: str,
        client_measurement: bytes,
        client_mac: bytes,
        probe_ok: dict[str, bool] | None = None,
        now: float | None = None,
    ) -> AttestationResult:
        now = time.time() if now is None else now
        st = self.state(session_id)
        ch = st.last_challenge
        if ch is None or ch.challenge_id != challenge_id:
            res = AttestationResult(ok=False, reason="unknown_or_stale_challenge")
            return self._record(st, res, tamper=False)

        latency_ms = (now - ch.issued_at) * 1000.0
        if now > ch.deadline:
            res = AttestationResult(
                ok=False, reason="deadline_exceeded", latency_ms=latency_ms
            )
            return self._record(st, res, tamper=False)

        expected = self.expected_mac(ch.nonce, client_measurement)
        if not hmac.compare_digest(expected, client_mac):
            res = AttestationResult(
                ok=False, reason="mac_mismatch", latency_ms=latency_ms,
                detail={"probe_ok": probe_ok or {}},
            )
            # A well-formed response with a bad MAC is tamper, not loss.
            return self._record(st, res, tamper=True)

        probe_ok = probe_ok or {}
        if probe_ok and not all(probe_ok.values()):
            res = AttestationResult(
                ok=False, reason="probe_failed", latency_ms=latency_ms,
                detail={"probe_ok": probe_ok},
            )
            return self._record(st, res, tamper=True)

        res = AttestationResult(ok=True, reason="ok", latency_ms=latency_ms, detail={"probe_ok": probe_ok})
        return self._record(st, res, tamper=False)

    def _record(self, st: SessionIntegrity, res: AttestationResult, *, tamper: bool) -> AttestationResult:
        st.history.append(res)
        if res.ok:
            st.successes += 1
            st.failures = 0
            st.attested = True
        else:
            st.failures += 1
            if st.failures >= self.cfg.untrusted_after_failures:
                st.untrusted = True
                st.attested = False
            if tamper:
                st.confirmed_tamper = True
                st.untrusted = True
                st.attested = False
        return res

    # -- lab helper: simulate a honest / dishonest client -----------------

    def simulate_honest_response(self, ch: Challenge, measurement: bytes | None = None) -> tuple[bytes, bytes]:
        measurement = measurement or os.urandom(32)
        mac = self.expected_mac(ch.nonce, measurement)
        return measurement, mac
