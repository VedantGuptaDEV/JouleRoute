"""Train the logistic-regression baseline router (3-class, class-weighted).

The regularization strength C is chosen on val by expected cost (cost matrix from config).
Saves artifacts/baseline/baseline.joblib (featurizer + model in one sklearn pipeline).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from router.train.common import BASELINE, artifacts_dir, classification_metrics, cost_matrix, load_labeled  # noqa: E402
from router.train.features import build_featurizer  # noqa: E402
from shared.config import load_config  # noqa: E402


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    args = parser.parse_args()

    train, val = load_labeled(cfg, "train"), load_labeled(cfg, "val")
    C = cost_matrix(cfg)
    results, best = [], None
    for c in args.c_grid:
        model = make_pipeline(build_featurizer(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=5000, random_state=cfg["seed"]))
        model.fit(train.text.tolist(), train.label.values)
        # predict_proba only has columns for classes seen in training; the 3-class layout matters.
        if list(model.classes_) != [0, 1, 2]:
            sys.exit(f"train split is missing classes (has {list(model.classes_)}); need more data")
        m = classification_metrics(model.predict_proba(val.text.tolist()), val.label.values, C)
        results.append({"C": c, **m})
        print(f"C={c:<6} val acc={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f} "
              f"expected cost={m['expected_cost']:.3f}")
        if best is None or m["expected_cost"] < best[1]["expected_cost"]:
            best = (model, m, c)

    out = artifacts_dir(cfg, BASELINE)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best[0], out / "baseline.joblib")
    (out / "train_log.json").write_text(json.dumps({"grid": results, "best_C": best[2]}, indent=2))

    print(f"\nBest C={best[2]}: val acc={best[1]['accuracy']:.3f}, macro-F1={best[1]['macro_f1']:.3f}, "
          f"expected cost={best[1]['expected_cost']:.3f}")
    print("Val confusion (rows=true, cols=pred; LITE/MID/FULL):", best[1]["confusion"])
    print(f"Saved {out / 'baseline.joblib'}")


if __name__ == "__main__":
    main()
