"""Temperature scaling for a trained router, fitted on the val split.

Finds a single T > 0 minimizing validation NLL of softmax(logits / T). Saves
artifacts/<name>/temperature.json, merges ECE before/after into results/calibration.json,
and writes a 10-bin reliability diagram to results/reliability_<name>.png.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from router.train.common import artifacts_dir, get_logits, load_labeled, results_dir, softmax  # noqa: E402
from shared.config import load_config  # noqa: E402

N_BINS = 10


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    # Optimize log T so T stays positive.
    res = minimize_scalar(lambda log_t: nll(logits, labels, float(np.exp(log_t))), bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    """Per-bin (confidence, accuracy, count) of the top prediction, and the ECE."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bins.append({"lo": lo, "hi": hi, "confidence": float(conf[mask].mean()),
                         "accuracy": float(correct[mask].mean()), "count": int(mask.sum())})
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
        else:
            bins.append({"lo": lo, "hi": hi, "confidence": None, "accuracy": None, "count": 0})
    return bins, float(ece)


def plot(before, after, ece_before, ece_after, temperature, name, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, bins, ece, title in [(axes[0], before, ece_before, "before (T=1)"),
                                 (axes[1], after, ece_after, f"after (T={temperature:.2f})")]:
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        acc = [b["accuracy"] or 0 for b in bins]
        ax.bar(centers, acc, width=1 / N_BINS, edgecolor="black", alpha=0.7, label="accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set_title(f"{name} {title}\nECE = {ece:.3f}")
        ax.set_xlabel("confidence (max probability)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="bert_tiny", help="artifact folder of the router (bert_tiny, minilm, baseline)")
    args = parser.parse_args()

    val = load_labeled(cfg, "val")
    labels = val.label.values
    logits = get_logits(cfg, args.name, val.text.tolist())
    temperature = fit_temperature(logits, labels)

    before, ece_before = reliability(softmax(logits), labels)
    after, ece_after = reliability(softmax(logits, temperature), labels)
    record = {"temperature": temperature, "fitted_on": "val", "n": len(val),
              "nll_before": nll(logits, labels, 1.0), "nll_after": nll(logits, labels, temperature),
              "ece_before": ece_before, "ece_after": ece_after,
              "bins_before": before, "bins_after": after}

    (artifacts_dir(cfg, args.name) / "temperature.json").write_text(
        json.dumps({"temperature": temperature, "fitted_on": "val"}, indent=2))
    out = results_dir(cfg)
    cal_path = out / "calibration.json"
    all_cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    all_cal[args.name] = record
    cal_path.write_text(json.dumps(all_cal, indent=2))
    plot(before, after, ece_before, ece_after, temperature, args.name, out / f"reliability_{args.name}.png")

    print(f"{args.name}: T = {temperature:.3f} (fitted on {len(val)} val queries)")
    print(f"  NLL {record['nll_before']:.4f} -> {record['nll_after']:.4f}")
    print(f"  ECE {ece_before:.4f} -> {ece_after:.4f}")
    print(f"  saved temperature.json, {cal_path.name}, reliability_{args.name}.png")


if __name__ == "__main__":
    main()
