"""Sweep the gate threshold tau and pick a risk-controlled floor tau_min (calib split only).

Gate: confidence >= tau -> predicted tier, else FULL. Per-query loss is 1 when the routed
tier answered wrongly but FULL answered correctly (accuracy lost vs. always-full). Raising
tau only moves queries to FULL, so loss is non-increasing in tau, which makes these valid:

  tau_min  smallest tau whose Clopper-Pearson (1 - delta) upper bound on calib loss is <= alpha,
           scanning from high tau to low and stopping at the first failure (fixed-sequence
           testing). Supports "accuracy loss <= alpha with probability >= 1 - delta".
  tau_crc  conformal risk control: smallest tau with (n * R(tau) + 1) / (n + 1) <= alpha,
           which bounds the *expected* loss.

Energy in the sweep uses PLACEHOLDER per-tier weights from config until measured joules exist.
Writes results/threshold_sweep.csv and artifacts/<name>/thresholds.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from router.train.common import (  # noqa: E402
    artifacts_dir, get_logits, load_labeled, load_temperature, results_dir, softmax,
)
from shared.config import load_config  # noqa: E402
from shared.tiers import TIER_KEYS, Tier  # noqa: E402


def gate(probs: np.ndarray, tau: float) -> np.ndarray:
    """Routed tier per query: argmax if its probability >= tau, else FULL."""
    pred = probs.argmax(axis=1)
    return np.where(probs.max(axis=1) >= tau, pred, int(Tier.FULL))


def routing_loss(routed: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """correct: [n, 3] 0/1 per tier. Loss 1 = routed tier wrong while FULL right."""
    routed_correct = correct[np.arange(len(routed)), routed]
    return ((routed_correct == 0) & (correct[:, Tier.FULL] == 1)).astype(int)


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) upper confidence bound on a binomial rate with k successes in n."""
    return 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))


def fixed_sequence_tau_min(taus: np.ndarray, losses: list[np.ndarray], alpha: float, delta: float):
    """Scan from the highest tau down; return the last tau before the first bound violation (None if the first fails)."""
    tau_min = None
    for tau, loss in sorted(zip(taus, losses), key=lambda p: -p[0]):
        if clopper_pearson_upper(int(loss.sum()), len(loss), delta) > alpha:
            break
        tau_min = float(tau)
    return tau_min


def crc_tau(taus: np.ndarray, losses: list[np.ndarray], alpha: float):
    ok = [float(t) for t, loss in zip(taus, losses) if (loss.sum() + 1) / (len(loss) + 1) <= alpha]
    return min(ok) if ok else None


def mode_presets(tau_min: float, top: float) -> dict[str, float]:
    """Saver = tau_min; balanced and performance at 1/3 and 2/3 of the way to the top of the sweep."""
    span = max(top - tau_min, 0.0)
    return {"saver": round(tau_min, 2), "balanced": round(tau_min + span / 3, 2),
            "performance": round(tau_min + 2 * span / 3, 2)}


def main():
    cfg = load_config()
    r = cfg["router"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="bert_tiny", help="router artifact folder (bert_tiny, minilm, baseline)")
    parser.add_argument("--alpha", type=float, default=r["alpha"], help="max allowed accuracy loss vs. always-full")
    parser.add_argument("--delta", type=float, default=r["delta"], help="allowed failure probability of the bound")
    args = parser.parse_args()

    calib = load_labeled(cfg, "calib")
    correct = calib[[f"{t}_correct" for t in TIER_KEYS]].values
    probs = softmax(get_logits(cfg, args.name, calib.text.tolist()), load_temperature(cfg, args.name))
    weights = np.array([r["energy_weights_placeholder"][t] for t in TIER_KEYS])
    full_acc = correct[:, Tier.FULL].mean()

    s = r["tau_sweep"]
    taus = np.round(np.arange(s["start"], s["stop"] + 1e-9, s["step"]), 2)
    rows, losses = [], []
    for tau in taus:
        routed = gate(probs, tau)
        loss = routing_loss(routed, correct)
        losses.append(loss)
        shares = np.bincount(routed, minlength=3) / len(routed)
        rows.append({
            "router": args.name, "tau": tau, "n": len(routed),
            **{f"share_{t}": shares[i] for i, t in enumerate(TIER_KEYS)},
            "fallback_rate": float(((probs.max(axis=1) < tau) & (probs.argmax(axis=1) != Tier.FULL)).mean()),
            "accuracy": float(correct[np.arange(len(routed)), routed].mean()),
            "accuracy_full": float(full_acc),
            "accuracy_loss": float(loss.mean()),
            "loss_ucb": clopper_pearson_upper(int(loss.sum()), len(loss), args.delta),
            "energy_rel_placeholder": float(weights[routed].sum() / (weights[Tier.FULL] * len(routed))),
        })
    sweep = pd.DataFrame(rows)

    sweep_path = results_dir(cfg) / "threshold_sweep.csv"
    if sweep_path.exists():
        old = pd.read_csv(sweep_path)
        sweep = pd.concat([old[old.router != args.name], sweep], ignore_index=True)
    sweep.to_csv(sweep_path, index=False)

    tau_min = fixed_sequence_tau_min(taus, losses, args.alpha, args.delta)
    tau_c = crc_tau(taus, losses, args.alpha)
    warning = None
    if tau_min is None:
        warning = (f"NO tau in the sweep satisfies alpha={args.alpha} at delta={args.delta} on {len(calib)} calib "
                   f"queries. tau_min is set to 1.0, which sends every query to FULL.")
        print("\n" + "!" * 80 + f"\nWARNING: {warning}\n" + "!" * 80)
        tau_min = 1.0
    presets = mode_presets(tau_min, float(taus.max())) if tau_min < 1.0 else \
        {"saver": 1.0, "balanced": 1.0, "performance": 1.0}
    thresholds = {
        "alpha": args.alpha, "delta": args.delta, "n_calib": len(calib),
        "tau_min": tau_min, "tau_crc": tau_c,
        "default_tau": presets["balanced"], "mode_presets": presets,
        "energy_weights": "placeholder", "warning": warning,
    }
    (artifacts_dir(cfg, args.name) / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    print(f"\n{args.name}: calib n={len(calib)}, always-full accuracy {full_acc:.3f}")
    show = sweep[(sweep.router == args.name) & sweep.tau.isin([0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99] + list(presets.values()))]
    print(show[["tau", "share_lite", "share_mid", "share_full", "fallback_rate", "accuracy_loss", "loss_ucb",
                "energy_rel_placeholder"]].round(3).to_string(index=False))
    print(f"\ntau_min (Clopper-Pearson, fixed-sequence) = {tau_min}   tau_crc = {tau_c}")
    print(f"presets = {presets}  (energy uses PLACEHOLDER weights {r['energy_weights_placeholder']})")


if __name__ == "__main__":
    main()
