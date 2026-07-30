"""Aegis CLI.

Commands:

  aegis train          bootstrap models from synthetic data
  aegis evaluate       score a saved model on a held-out synthetic set
  aegis simulate       write one synthetic session capture for inspection
  aegis serve          launch the FastAPI adjudication service
  aegis explain        print an explanation for a scored synthetic lobby
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

    p_serve = sub.add_parser("serve", help="run the adjudication API")
    p_serve.add_argument("--model-dir", type=str, default="models")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)

    p_exp = sub.add_parser("explain", help="train-free demo explanation on a tiny lobby")
    p_exp.add_argument("--model-dir", type=str, default="models")
    p_exp.add_argument("--seed", type=int, default=7)

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
        from .train.dataset import generate_dataset  # noqa: F401
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
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
        return 0

    if args.cmd == "serve":
        import uvicorn
        from .service import create_app

        app = create_app(model_dir=args.model_dir)
        uvicorn.run(app, host=args.host, port=args.port)
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
