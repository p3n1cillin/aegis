"""Aegis CLI.

Commands:

  aegis train          bootstrap models from synthetic data
  aegis evaluate       score a saved model on a held-out synthetic set
  aegis simulate       write one synthetic session capture for inspection
  aegis serve          launch the FastAPI adjudication service
  aegis ingest         live plugin TCP ingest (captures + score)
  aegis explain        print an explanation for a scored synthetic lobby
  aegis online-refresh fold human review labels into calibrators / novelty
  aegis audit           read the verdict audit log (JSONL)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aegis", description="Aegis adaptive anti-cheat")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train ensemble on synthetic data")
    p_train.add_argument("--sessions", type=int, default=48)
    p_train.add_argument("--duration", type=float, default=75.0)
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.add_argument("--out", type=str, default="models")
    p_train.add_argument("--skip-temporal", action="store_true")
    p_train.add_argument("--device", type=str, default="cpu")

    p_eval = sub.add_parser("evaluate", help="evaluate a saved ensemble")
    p_eval.add_argument("--model-dir", type=str, default="models")
    p_eval.add_argument("--sessions", type=int, default=24)
    p_eval.add_argument("--duration", type=float, default=60.0)
    p_eval.add_argument("--seed", type=int, default=4242)
    p_eval.add_argument("--stealth-curve", action="store_true")

    p_sim = sub.add_parser("simulate", help="generate one synthetic session")
    p_sim.add_argument("--out", type=str, default="data/sample_session")
    p_sim.add_argument("--duration", type=float, default=60.0)
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.add_argument("--families", type=str, default="aimbot_legit,clean,wallhack,clean,clean,norecoil")
    p_sim.add_argument("--aegis-capture", action="store_true",
                       help="also write a plugin-protocol .aegis capture")

    p_serve = sub.add_parser("serve", help="run the adjudication API")
    p_serve.add_argument("--model-dir", type=str, default="models")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)

    p_ing = sub.add_parser("ingest", help="live CS2 plugin TCP ingest")
    p_ing.add_argument("--model-dir", type=str, default="models")
    p_ing.add_argument("--capture-dir", type=str, default="data/captures")
    p_ing.add_argument("--host", type=str, default="0.0.0.0")
    p_ing.add_argument("--port", type=int, default=9100)

    p_exp = sub.add_parser("explain", help="train-free demo explanation on a tiny lobby")
    p_exp.add_argument("--model-dir", type=str, default="models")
    p_exp.add_argument("--seed", type=int, default=7)

    p_on = sub.add_parser("online-refresh", help="refresh calibrators/novelty from review labels")
    p_on.add_argument("--model-dir", type=str, default="models")
    p_on.add_argument("--labels", type=str, required=True, help="JSONL review label store")
    p_on.add_argument("--out", type=str, default=None, help="defaults to --model-dir")

    p_audit = sub.add_parser("audit", help="read the verdict audit log")
    p_audit.add_argument("--log", type=str, default="audit/verdicts.jsonl")
    p_audit.add_argument("--tail", type=int, default=20)
    p_audit.add_argument("--json", action="store_true", help="emit raw JSON lines")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        from .core.config import AegisConfig, ModelConfig
        from .train.train import bootstrap_train

        cfg = AegisConfig(model_dir=args.out, model=ModelConfig(device=args.device))
        print(f"[aegis] simulating {args.sessions} sessions @ {args.duration}s ...")
        ens, report = bootstrap_train(
            n_sessions=args.sessions, duration_s=args.duration, seed=args.seed,
            out_dir=args.out, cfg=cfg, skip_temporal=args.skip_temporal,
        )
        print(json.dumps({k: report[k] for k in report if k != "temporal_history"}, indent=2, default=str))
        if "test" in report:
            t = report["test"]
            print(f"[aegis] test AUC={t.get('auc'):.3f} AP={t.get('ap'):.3f} ECE={t.get('ece'):.3f}")
            for op in t.get("operating_points", []):
                print(f"  {op['name']:12s} thr={op['threshold']:.3f}  "
                      f"FPR={op['fpr']:.4f}  TPR={op['tpr']:.3f}  prec={op['precision']:.3f}")
            if t.get("per_family_recall"):
                print("[aegis] recall@review by family:")
                for fam, r in sorted(t["per_family_recall"].items()):
                    print(f"  {fam:16s} {r:.3f}")
        print(f"[aegis] model saved to {args.out}/ (spec {ens.digest})")
        return 0

    if args.cmd == "evaluate":
        from .models.ensemble import CalibratedEnsemble
        from .train.dataset import generate_dataset
        from .train.evaluate import detection_vs_stealth, evaluate_ensemble
        from .core.config import AegisConfig

        ens = CalibratedEnsemble.load(args.model_dir)
        ds = generate_dataset(
            n_sessions=args.sessions, duration_s=args.duration, seed=args.seed, seq_len=128,
        )
        ev = evaluate_ensemble(ens, ds, cfg=AegisConfig(model_dir=args.model_dir))
        print(json.dumps(ev.to_dict(), indent=2))
        if args.stealth_curve:
            curve = detection_vs_stealth(ens, n_per_level=4, duration_s=45.0)
            print("[aegis] detection-vs-stealth:")
            print(json.dumps(curve, indent=2))
        return 0

    if args.cmd == "simulate":
        from .train.simulate import simulate_session
        import numpy as np

        families = [f.strip() for f in args.families.split(",") if f.strip()]
        sess, labels = simulate_session(
            session_id=f"cli_{args.seed}", cheat_families=families,
            duration_s=args.duration, seed=args.seed,
        )
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "ticks.npz", ticks=sess.ticks)
        meta = {
            "session_id": sess.session_id,
            "labels": labels,
            "n_ticks": int(sess.ticks.size),
            "n_shots": len(sess.shots),
            "n_kills": len(sess.kills),
            "n_vis": len(sess.visibility),
            "duration": sess.duration(),
            "players": {
                str(s): {"player_id": p.player_id, "team": p.team}
                for s, p in sess.players.items()
            },
        }
        if args.aegis_capture:
            from .adapters.cs2.capture import session_to_frames, write_capture
            cap = write_capture(out / f"{sess.session_id}.aegis", session_to_frames(sess))
            meta["aegis_capture"] = str(cap)
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
        return 0

    if args.cmd == "serve":
        import os

        import uvicorn

        from .core.config import AegisConfig
        from .service import create_app

        cfg = AegisConfig(model_dir=args.model_dir)
        cfg.api_key = os.environ.get("AEGIS_API_KEY", "")
        if args.host not in ("127.0.0.1", "localhost", "::1") and not cfg.api_key:
            print(
                f"[aegis] WARNING: serving on {args.host} (not loopback) with no "
                "AEGIS_API_KEY set -- /v1/* routes are reachable by anyone who can "
                "route to this host.",
                file=sys.stderr,
            )
        app = create_app(model_dir=args.model_dir, config=cfg)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.cmd == "ingest":
        from .service.ingest import serve_ingest

        serve_ingest(
            host=args.host, port=args.port,
            model_dir=args.model_dir, capture_dir=args.capture_dir,
        )
        return 0

    if args.cmd == "explain":
        from .models.ensemble import CalibratedEnsemble
        from .train.simulate import simulate_session
        from .features.pipeline import extract_session_features
        from .train.dataset import SIM_CAPS
        from .fusion.explain import explain_score

        path = Path(args.model_dir)
        if not (path / "ensemble.json").exists():
            print("no model found; run `aegis train` first", file=sys.stderr)
            return 1
        ens = CalibratedEnsemble.load(path)
        sess, labels = simulate_session(
            session_id="explain_demo",
            cheat_families=["blatant", "clean", "aimbot_legit", "clean"],
            duration_s=45.0, seed=args.seed,
        )
        feats = extract_session_features(sess, set(SIM_CAPS), spec=ens.spec)
        for f in feats:
            sc = ens.score_player(
                session_id=f.session_id, slot=f.slot, player_id=f.player_id,
                features=f.values,
            )
            expl = explain_score(sc, features=f, tabular=ens.bundle.tabular, spec=ens.spec)
            print("=" * 60)
            print(f"label={labels.get(f.slot)}  player={f.player_id}")
            print(expl.as_text())
        return 0

    if args.cmd == "online-refresh":
        from .models.ensemble import CalibratedEnsemble
        from .train.online import OnlineStore, refresh_from_reviews

        ens = CalibratedEnsemble.load(args.model_dir)
        store = OnlineStore.load(args.labels)
        ens, report = refresh_from_reviews(
            ens, store, out_dir=args.out or args.model_dir,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    if args.cmd == "audit":
        from .integrity.audit import AuditLog

        records = AuditLog(args.log).tail(args.tail)
        if not records:
            print(f"[aegis] no audit records in {args.log}")
            return 0
        for r in records:
            if args.json:
                print(r.to_json())
            else:
                print(
                    f"{r.ts}  {r.verdict:<14} shadow={str(r.shadow):<5} "
                    f"p={r.p_fused:.3f} conf_p={r.conformal_p:.2e} "
                    f"eng={r.n_engagements:<4d} {r.player_id} ({r.session_id})"
                )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
