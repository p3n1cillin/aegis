"""Model artefact signing (HANDOFF "Next work" item 8).

A tampered ``tabular.joblib`` is the cleanest attack on the whole platform:
no telemetry ever needs to look wrong, every verdict just comes out however
the attacker wants. These helpers provide HMAC-SHA256 signatures over the
deterministic artefact file set so an operator who holds the key can detect
it.

Design notes:
  * HMAC-SHA256 via ``cryptography`` -- no new dependency, and unlike
    asymmetric signatures this does not pretend to solve key distribution;
    the honest threat model is "operator notices tampering", not "attacker
    never held the machine".
  * Signature covers file *contents* of every artefact except the signature
    file itself, sorted by relative path for determinism.
  * Fail-closed wiring: if ``AEGIS_MODEL_KEY`` is set in the environment,
    ``CalibratedEnsemble.save`` signs automatically and ``load`` verifies,
    refusing to serve from a directory that fails verification. With no key
    set nothing changes (lab workflow stays friction-free).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

SIGNATURE_FILE = "SIG.json"


def _artefact_files(model_dir: Path) -> list[Path]:
    return sorted(
        p for p in model_dir.rglob("*")
        if p.is_file() and p.name != SIGNATURE_FILE
    )


def _canonical_digest(model_dir: Path) -> str:
    """SHA-256 over (relative path, length, content) triples, path-sorted."""
    h = hashlib.sha256()
    for p in _artefact_files(model_dir):
        rel = p.relative_to(model_dir).as_posix()
        data = p.read_bytes()
        h.update(rel.encode("utf-8"))
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def sign_model_dir(model_dir: str | Path, key: bytes) -> dict:
    """Write ``SIG.json`` covering every artefact file under ``model_dir``."""
    import datetime

    model_dir = Path(model_dir)
    digest = _canonical_digest(model_dir)
    sig = hmac.new(key, digest.encode("ascii"), hashlib.sha256).hexdigest()
    payload = {
        "algorithm": "HMAC-SHA256",
        "digest": digest,
        "signature": sig,
        # ISO ts is metadata only; it is not part of what gets verified.
        "signed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "files": [p.relative_to(model_dir).as_posix() for p in _artefact_files(model_dir)],
    }
    (model_dir / SIGNATURE_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_model_dir(model_dir: str | Path, key: bytes) -> dict:
    """Verify ``SIG.json`` against the artefacts on disk.

    Raises ``PermissionError`` (not ValueError -- tampering is a security
    event, not bad input) when the signature is missing, was made with a
    different key, or any covered file has changed since signing.
    """
    model_dir = Path(model_dir)
    sig_path = model_dir / SIGNATURE_FILE
    if not sig_path.exists():
        raise PermissionError(
            f"{sig_path} missing: cannot verify model artefacts under {model_dir}"
        )
    payload = json.loads(sig_path.read_text(encoding="utf-8"))
    digest = _canonical_digest(model_dir)
    expected = hmac.new(key, digest.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, payload.get("signature", "")):
        raise PermissionError(
            f"model artefact signature mismatch under {model_dir}: "
            "files changed since signing, or the key is wrong"
        )
    return payload
