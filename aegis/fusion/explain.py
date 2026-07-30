"""Explainability for reviewers and appeals.

A ban that cannot be explained is a ban that will not survive an appeal, and
a detector that cannot explain itself cannot be red-teamed. This module turns
a ``Score`` + tabular importances + top engagement LLRs into a structured
rationale a human can read in under a minute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..features.pipeline import FeatureSpec, SessionFeatures
from ..models.ensemble import Score
from ..models.tabular import TabularGBDT


@dataclass
class Explanation:
    player_id: str
    session_id: str
    summary: str
    top_features: list[tuple[str, float, float]]  # (name, value, importance)
    top_engagements: list[dict]
    modality_breakdown: dict[str, float]
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "session_id": self.session_id,
            "summary": self.summary,
            "top_features": [
                {"name": n, "value": v, "importance": i} for n, v, i in self.top_features
            ],
            "top_engagements": self.top_engagements,
            "modality_breakdown": self.modality_breakdown,
            "caveats": self.caveats,
        }

    def as_text(self) -> str:
        lines = [self.summary, "", "Modality breakdown:"]
        for k, v in self.modality_breakdown.items():
            lines.append(f"  {k}: {v:.3f}")
        lines.append("", "Top contributing features:")
        for n, v, i in self.top_features[:8]:
            lines.append(f"  {n:40s}  value={v: .4f}  importance={i:.4f}")
        if self.top_engagements:
            lines.append("", "Most suspicious engagements:")
            for e in self.top_engagements[:5]:
                lines.append(
                    f"  #{e['index']}: p={e['p']:.3f} llr={e['llr']:.2f}"
                )
        if self.caveats:
            lines.append("", "Caveats:")
            for c in self.caveats:
                lines.append(f"  - {c}")
        return "\n".join(lines)


def explain_score(
    score: Score,
    features: SessionFeatures | None = None,
    tabular: TabularGBDT | None = None,
    spec: FeatureSpec | None = None,
    top_k: int = 10,
) -> Explanation:
    caveats = []
    top_feats: list[tuple[str, float, float]] = []

    if features is not None and tabular is not None and tabular.feature_importance_:
        imp = tabular.feature_importance_
        vals = features.as_dict()
        ranked = sorted(imp.items(), key=lambda kv: abs(kv[1]), reverse=True)
        for name, importance in ranked[:top_k]:
            top_feats.append((name, float(vals.get(name, float("nan"))), float(importance)))
    elif features is not None and spec is not None:
        # Fall back to |z|-ish ranking by absolute feature value after nan-fill.
        for name, val in zip(spec.columns, features.values):
            if np.isfinite(val):
                top_feats.append((name, float(val), abs(float(val))))
        top_feats.sort(key=lambda x: x[2], reverse=True)
        top_feats = top_feats[:top_k]
        caveats.append("feature importances unavailable; ranked by absolute value")
    else:
        caveats.append("no per-feature breakdown available for this score")

    eng = [
        {"index": i, "p": float(p), "llr": float(llr)}
        for i, (p, llr) in enumerate(zip(score.engagement_ps, score.engagement_llrs))
    ]
    eng.sort(key=lambda d: d["llr"], reverse=True)

    modality = {
        "tabular": score.p_tabular,
        "temporal": score.p_temporal,
        "novelty": score.p_novelty,
        "fused": score.p_fused,
    }

    if score.p_novelty > 0.7 and score.p_tabular < 0.55:
        caveats.append(
            "high novelty with moderate supervised score -- possible novel cheat "
            "family or unusual legitimate style; prefer review over auto-ban"
        )
    if score.meta.get("n_engagements", 0) < 8:
        caveats.append("few engagements in this session; posterior is prior-dominated")
    if not score.engagement_ps:
        caveats.append("no engagement tensors scored; temporal channel silent")

    summary = (
        f"Player {score.player_id} in session {score.session_id}: "
        f"fused P(cheat)={score.p_fused:.3f}, conformal p={score.conformal_p:.2e}, "
        f"{len(score.engagement_ps)} engagements scored."
    )
    return Explanation(
        player_id=score.player_id,
        session_id=score.session_id,
        summary=summary,
        top_features=top_feats,
        top_engagements=eng[:10],
        modality_breakdown=modality,
        caveats=caveats,
    )
