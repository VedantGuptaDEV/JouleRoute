"""
router/scripts/train_router.py
Fine-tunes BERT-Tiny on the generated dataset, calibrates tau, writes artifacts.

Run:
    python router/scripts/train_router.py --data router/data/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import sys
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.config import load_config as load_cfg

LABEL_NAMES = ["lite", "mid", "full"]
NUM_LABELS  = 3


# ── helpers ───────────────────────────────────────────────────────────────────

def _metrics(preds: list[int], labels: list[int], cost_matrix: list[list[int]]) -> dict:
    n = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    # macro F1
    f1s = []
    for c in range(NUM_LABELS):
        tp = sum(p == c and l == c for p, l in zip(preds, labels))
        fp = sum(p == c and l != c for p, l in zip(preds, labels))
        fn = sum(p != c and l == c for p, l in zip(preds, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    # confusion
    conf = [[0]*NUM_LABELS for _ in range(NUM_LABELS)]
    for p, l in zip(preds, labels):
        conf[l][p] += 1
    # expected cost
    exp_cost = sum(cost_matrix[l][p] for p, l in zip(preds, labels)) / n
    return {
        "accuracy":      correct / n,
        "macro_f1":      float(np.mean(f1s)),
        "expected_cost": exp_cost,
        "confusion":     conf,
    }


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _calibrate_temperature(logits: list[np.ndarray], labels: list[int],
                            grid: tuple = (0.3, 5.0, 100)) -> float:
    """Grid search for temperature scaling that minimises NLL on calib set."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(grid[0], grid[1], grid[2]):
        nll = 0.0
        for z, l in zip(logits, labels):
            p = _softmax(z / t)
            nll -= math.log(max(p[l], 1e-9))
        nll /= len(labels)
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def _select_tau(probs_calib: list[np.ndarray], labels: list[int],
                alpha: float, delta: float,
                sweep_start: float, sweep_stop: float, sweep_step: float,
                cost_matrix: list[list[int]]) -> tuple[float | None, float]:
    """
    Risk-control threshold selection (§5.A).
    Returns (tau_crc, tau_min).
    tau_min = smallest tau where the accuracy-loss bound holds at (alpha, delta).
    If no tau satisfies, returns (None, 1.0).
    """
    n = len(labels)
    taus = np.arange(sweep_start, sweep_stop + sweep_step / 2, sweep_step)
    valid: list[float] = []

    for tau in taus:
        preds = []
        for p in probs_calib:
            pred = int(p.argmax())
            conf = float(p.max())
            preds.append(pred if conf >= tau else NUM_LABELS - 1)  # fallback = FULL
        # harmful downgrade rate: routed to cheaper tier but FULL would be right
        # We use accuracy loss as the proxy: fraction wrong
        acc_loss = sum(p != l for p, l in zip(preds, labels)) / n
        # Hoeffding bound: P(true_loss > alpha) <= exp(-2n*(alpha - empirical)^2)
        # Invert: accept if empirical + sqrt(log(1/delta) / (2n)) <= alpha
        margin = math.sqrt(math.log(1.0 / delta) / (2 * n))
        if acc_loss + margin <= alpha:
            valid.append(float(tau))

    if not valid:
        log.warning(
            "NO tau satisfies alpha=%.2f at delta=%.2f on %d calib queries. "
            "tau_min set to 1.0 (sends everything to FULL).", alpha, delta, n
        )
        return None, 1.0

    tau_min = min(valid)
    tau_crc = tau_min  # use the most permissive valid tau as the default
    log.info("tau_crc=%.2f  tau_min=%.2f  (from %d valid taus)", tau_crc, tau_min, len(valid))
    return tau_crc, tau_min


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Train BERT-Tiny router")
    ap.add_argument("--data",       default="router/data/dataset.jsonl")
    ap.add_argument("--artifacts",  default=None,
                    help="output dir (default: router.artifacts_dir from config.yaml)")
    ap.add_argument("--epochs",     type=int,   default=8)
    ap.add_argument("--lr",         type=float, default=2e-5)
    ap.add_argument("--batch",      type=int,   default=16)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    cfg     = load_cfg()
    rcfg    = cfg.get("router", {})
    art_dir = Path(args.artifacts or rcfg.get("artifacts_dir", "router/artifacts/bert_tiny"))
    art_dir.mkdir(parents=True, exist_ok=True)

    base_model  = rcfg.get("base_model", "google/bert_uncased_L-2_H-128_A-2")
    max_length  = int(rcfg.get("max_length", 64))
    splits_cfg  = rcfg.get("splits", {"train": 0.60, "val": 0.10, "calib": 0.15, "test": 0.15})
    cost_matrix = rcfg.get("cost_matrix", [[0,1,2],[4,0,1],[6,4,0]])
    alpha       = float(rcfg.get("alpha", 0.03))
    delta       = float(rcfg.get("delta", 0.05))
    tau_sweep   = rcfg.get("tau_sweep", {"start": 0.30, "stop": 0.99, "step": 0.01})
    seed        = args.seed

    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(4)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    # ── load data ─────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Dataset not found: %s", data_path)
        raise SystemExit(1)

    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.shuffle(rows)
    n = len(rows)
    log.info("Loaded %d rows from %s", n, data_path)

    # split
    t_frac = splits_cfg["train"]
    v_frac = splits_cfg["val"]
    c_frac = splits_cfg["calib"]
    # test gets the rest
    i_v = int(n * t_frac)
    i_c = i_v + int(n * v_frac)
    i_t = i_c + int(n * c_frac)

    train_rows = rows[:i_v]
    val_rows   = rows[i_v:i_c]
    calib_rows = rows[i_c:i_t]
    test_rows  = rows[i_t:]

    log.info("Split: train=%d  val=%d  calib=%d  test=%d",
             len(train_rows), len(val_rows), len(calib_rows), len(test_rows))

    # ── tokenizer + model ──────────────────────────────────────────────────────
    log.info("Loading tokenizer and model from %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )

    class QueryDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            return self.rows[i]["text"], self.rows[i]["label"]

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    train_loader = DataLoader(QueryDataset(train_rows), batch_size=args.batch,
                              shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(QueryDataset(val_rows),   batch_size=32,
                              shuffle=False, collate_fn=collate)

    # ── cost-sensitive loss ────────────────────────────────────────────────────
    lam = float(rcfg.get("cost_lambda", 0.5))
    cost_t = torch.tensor(cost_matrix, dtype=torch.float32)

    def cost_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce   = nn.functional.cross_entropy(logits, labels)
        prob = torch.softmax(logits, dim=-1)                # (B, 3)
        exp_cost = (prob * cost_t[labels]).sum(dim=-1).mean()
        return (1 - lam) * ce + lam * exp_cost

    # ── training loop ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs,
    )

    best_f1   = -1.0
    best_state = None
    history   = []
    t_train   = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels_b = batch.pop("labels")
            optimizer.zero_grad()
            out  = model(**batch)
            loss = cost_loss(out.logits, labels_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # ── val ───────────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels, all_logits = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                labels_b = batch.pop("labels")
                out = model(**batch)
                preds = out.logits.argmax(dim=-1).tolist()
                all_preds.extend(preds)
                all_labels.extend(labels_b.tolist())
                all_logits.extend(out.logits.float().numpy())

        m = _metrics(all_preds, all_labels, cost_matrix)
        avg_loss = total_loss / len(train_loader)
        log.info(
            "Epoch %d/%d  loss=%.4f  acc=%.3f  f1=%.3f  cost=%.3f",
            epoch, args.epochs, avg_loss, m["accuracy"], m["macro_f1"], m["expected_cost"]
        )
        history.append({
            "epoch": epoch, "train_loss": avg_loss,
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "expected_cost": m["expected_cost"], "confusion": m["confusion"],
        })
        if m["macro_f1"] > best_f1:
            best_f1    = m["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = m

    train_seconds = time.perf_counter() - t_train
    log.info("Training done in %.1f s. Best epoch=%d  f1=%.3f", train_seconds, best_epoch, best_f1)

    # ── restore best weights ───────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    # ── temperature calibration on calib set ──────────────────────────────────
    log.info("Calibrating temperature on %d calib queries ...", len(calib_rows))
    calib_logits, calib_labels = [], []
    calib_loader = DataLoader(QueryDataset(calib_rows), batch_size=32,
                              shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for batch in calib_loader:
            labels_b = batch.pop("labels")
            out = model(**batch)
            calib_logits.extend(out.logits.float().numpy())
            calib_labels.extend(labels_b.tolist())

    temperature = _calibrate_temperature(calib_logits, calib_labels)
    log.info("Temperature = %.4f", temperature)

    calib_probs = [_softmax(z / temperature) for z in calib_logits]

    # ── tau selection ──────────────────────────────────────────────────────────
    tau_crc, tau_min = _select_tau(
        calib_probs, calib_labels, alpha, delta,
        tau_sweep["start"], tau_sweep["stop"], tau_sweep["step"],
        cost_matrix,
    )

    # ── test evaluation ────────────────────────────────────────────────────────
    if test_rows:
        log.info("Evaluating on %d test queries ...", len(test_rows))
        test_loader = DataLoader(QueryDataset(test_rows), batch_size=32,
                                 shuffle=False, collate_fn=collate)
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                labels_b = batch.pop("labels")
                out = model(**batch)
                logits_np = out.logits.float().numpy()
                for z, l in zip(logits_np, labels_b.tolist()):
                    p = _softmax(z / temperature)
                    pred = int(p.argmax())
                    conf = float(p.max())
                    # apply tau_min (use 0.5 if tau_min=1.0 to allow routing)
                    effective_tau = tau_min if tau_min < 1.0 else 0.5
                    test_preds.append(pred if conf >= effective_tau else NUM_LABELS - 1)
                    test_labels.append(l)
        test_m = _metrics(test_preds, test_labels, cost_matrix)
        log.info("Test: acc=%.3f  f1=%.3f  cost=%.3f",
                 test_m["accuracy"], test_m["macro_f1"], test_m["expected_cost"])
    else:
        test_m = {}

    # ── save artifacts ────────────────────────────────────────────────────────
    log.info("Saving artifacts to %s ...", art_dir)

    # model + tokenizer
    model.save_pretrained(art_dir)
    tokenizer.save_pretrained(art_dir)

    # temperature
    (art_dir / "temperature.json").write_text(
        json.dumps({"temperature": round(temperature, 6)}, indent=2)
    )

    # thresholds
    warning = None
    if tau_crc is None:
        warning = (
            f"NO tau satisfies alpha={alpha} at delta={delta} on {len(calib_rows)} calib queries. "
            f"tau_min set to 1.0 (sends everything to FULL)."
        )
    thresholds = {
        "alpha":    alpha,
        "delta":    delta,
        "n_calib":  len(calib_rows),
        "tau_min":  tau_min,
        "tau_crc":  tau_crc,
        "default_tau": tau_crc if tau_crc is not None else 1.0,
        "mode_presets": {
            "saver":       min(tau_min + 0.15, 0.99) if tau_crc is not None else 1.0,
            "balanced":    tau_crc if tau_crc is not None else 1.0,
            "performance": max((tau_crc or 1.0) - 0.10, tau_min) if tau_crc is not None else 1.0,
        },
        "energy_weights": "placeholder",
    }
    if warning:
        thresholds["warning"] = warning
    (art_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # train log
    train_log = {
        "model":         base_model,
        "lam":           lam,
        "cost_matrix":   cost_matrix,
        "max_length":    max_length,
        "n_train":       len(train_rows),
        "n_val":         len(val_rows),
        "n_calib":       len(calib_rows),
        "n_test":        len(test_rows),
        "best": {
            "epoch":   best_epoch,
            "metrics": best_metrics,
        },
        "test_metrics": test_m,
        "temperature":  round(temperature, 6),
        "tau_crc":      tau_crc,
        "tau_min":      tau_min,
        "history":      history,
        "train_seconds": round(train_seconds, 1),
    }
    (art_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))

    log.info("Done.")
    log.info("  Val  acc=%.3f  f1=%.3f", best_metrics["accuracy"], best_metrics["macro_f1"])
    if test_m:
        log.info("  Test acc=%.3f  f1=%.3f", test_m["accuracy"], test_m["macro_f1"])
    log.info("  tau_crc=%.2f  tau_min=%.2f  temperature=%.4f",
             tau_crc or 1.0, tau_min, temperature)
    log.info("")
    log.info("Next: update config.yaml  tau_override: null  (let thresholds.json drive routing)")


if __name__ == "__main__":
    main()
