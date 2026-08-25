"""Verdict audit log (HANDOFF "Next work" item 8).

Every adjudication -- including shadow verdicts that ban nobody -- appends one
JSONL line here. Purpose: an operator decision (a ban, a threshold change, a
shadow-mode flip) must be reconstructible from what the system actually
computed at the time, not from whatever the current models would say
retroactively. Append-only on purpose; this is evidence, not state.

Deliberately plain JSONL rather than sqlite/whatever: operators tail it, grep
it, and ship it with existing log tooling.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class AuditRecord:
    ts: str
    session_id: str
    player_id: str
    verdict: str  # clear | watch | review | ban | integrity_ban
    shadow: bool
    p_fused: float
    conformal_p: float
    reasons: list[str] = field(default_factory=list)
    model_digest: str = ""
    n_engagements: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class AuditLog:
    """Append-only JSONL audit trail. Thread-safe (ingest + API can race)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: AuditRecord) -> None:
        line = record.to_json()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def tail(self, n: int = 20) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[AuditRecord] = []
        for line in lines[-max(0, n):]:
            if not line.strip():
                continue
            raw = json.loads(line)
            out.append(AuditRecord(**raw))
        return out


def make_audit_record(
    *,
    session_id: str,
    player_id: str,
    verdict,
    score,
    shadow: bool,
    model_digest: str = "",
) -> AuditRecord:
    """Build an AuditRecord from live adjudication outputs.

    ``verdict`` is a ``fusion.policy.Verdict``; ``score`` a
    ``models.ensemble.Score``. Kept dependency-light via duck typing so tests
    can pass stand-ins.
    """
    return AuditRecord(
        ts=_now_iso(),
        session_id=session_id,
        player_id=player_id,
        verdict=str(getattr(verdict.band, "value", verdict.band)),
        shadow=bool(shadow),
        p_fused=float(score.p_fused),
        conformal_p=float(score.conformal_p),
        reasons=list(getattr(verdict, "reasons", []) or []),
        model_digest=model_digest,
        n_engagements=int(score.meta.get("n_engagements", 0)),
    )
