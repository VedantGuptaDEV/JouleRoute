"""Fine-tune BERT-Tiny (or MiniLM via --model) as the 3-class router.

Loss = class-weighted cross-entropy + lambda * expected cost, where expected cost is
sum_j p_j * C[y, j] with the cost matrix from config (under-routing costs more than
over-routing). Tries every learning rate in the grid and keeps the checkpoint with the
lowest validation expected cost (not accuracy). Saves to artifacts/<name>/.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from router.train.common import N_TIERS, artifacts_dir, classification_metrics, cost_matrix, load_labeled  # noqa: E402
from shared.config import load_config  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse label frequency, normalized so a balanced set gets weight 1 per class."""
    counts = np.bincount(labels, minlength=N_TIERS).astype(np.float64)
    w = np.where(counts > 0, len(labels) / (N_TIERS * np.maximum(counts, 1)), 1.0)
    return torch.tensor(w, dtype=torch.float32)


def cost_sensitive_loss(logits, labels, weights, C, lam):
    ce = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    exp_cost = (logits.softmax(dim=1) * C[labels]).sum(dim=1).mean()
    return ce + lam * exp_cost


def encode(tok, texts, labels, max_length):
    enc = tok(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long))


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    return np.concatenate([model(input_ids=ids.to(device), attention_mask=mask.to(device))
                           .logits.softmax(dim=1).cpu().numpy() for ids, mask, _ in loader])


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda, but this PyTorch build has no CUDA (see TRAINING.md)")
    return torch.device(name)


def train_one_lr(args, cfg, tok, train_ds, val_ds, val_labels, lr, C_np):
    set_seed(cfg["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=N_TIERS).to(args.device)
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=256)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)
    weights = class_weights(train_ds.tensors[2].numpy()).to(args.device)
    C = torch.tensor(C_np, dtype=torch.float32, device=args.device)

    history, best, since_best = [], None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(args.device), mask.to(args.device), y.to(args.device)
            loss = cost_sensitive_loss(model(input_ids=ids, attention_mask=mask).logits, y, weights, C, args.lam)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            running += loss.item() * len(y)
        m = classification_metrics(predict_probs(model, val_loader, args.device), val_labels, C_np)
        history.append({"lr": lr, "epoch": epoch, "train_loss": running / len(train_ds), **m})
        print(f"  lr={lr:g} epoch {epoch}: train loss {running / len(train_ds):.4f} | val acc {m['accuracy']:.3f} "
              f"macro-F1 {m['macro_f1']:.3f} expected cost {m['expected_cost']:.4f}")
        if best is None or m["expected_cost"] < best["metrics"]["expected_cost"]:
            best = {"lr": lr, "epoch": epoch, "metrics": m,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            since_best = 0
        else:
            since_best += 1
            if epoch >= args.min_epochs and since_best >= args.patience:
                break
    return model, best, history


def main():
    cfg = load_config()
    r = cfg["router"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=r["base_model"], help=f"HF model id (upper bound: {r['upper_model']})")
    parser.add_argument("--name", default=None, help="artifact folder name (default: bert_tiny or minilm)")
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-4, 5e-5])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2, help="epochs without val improvement before stopping")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lam", type=float, default=r["cost_lambda"], help="weight of the expected-cost term")
    parser.add_argument("--cost-matrix", type=json.loads, default=r["cost_matrix"],
                        help="JSON 3x3 C[true][pred], e.g. '[[0,1,2],[4,0,1],[6,4,0]]'")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="auto = CUDA if this PyTorch build has it, else CPU")
    args = parser.parse_args()
    args.device = pick_device(args.device)
    name = args.name or ("minilm" if "MiniLM" in args.model else "bert_tiny")
    torch.set_num_threads(args.threads)
    C_np = np.asarray(args.cost_matrix, dtype=np.float64)

    train, val = load_labeled(cfg, "train"), load_labeled(cfg, "val")
    tok = AutoTokenizer.from_pretrained(args.model)
    train_ds = encode(tok, train.text.tolist(), train.label.values, r["max_length"])
    val_ds = encode(tok, val.text.tolist(), val.label.values, r["max_length"])
    print(f"Training {args.model} on {len(train)} queries (val {len(val)}) on {args.device}; "
          f"label counts train={np.bincount(train.label, minlength=3).tolist()}")

    t0 = time.perf_counter()
    best, history = None, []
    for lr in args.lrs:
        model, run_best, run_history = train_one_lr(args, cfg, tok, train_ds, val_ds, val.label.values, lr, C_np)
        history += run_history
        if best is None or run_best["metrics"]["expected_cost"] < best["metrics"]["expected_cost"]:
            best = run_best
            best_model = model

    out = artifacts_dir(cfg, name)
    out.mkdir(parents=True, exist_ok=True)
    best_model = best_model.cpu()
    best_model.load_state_dict(best["state"])
    best_model.save_pretrained(out)
    tok.save_pretrained(out)
    log = {"model": args.model, "lam": args.lam, "cost_matrix": args.cost_matrix, "max_length": r["max_length"],
           "best": {k: v for k, v in best.items() if k != "state"}, "history": history,
           "train_seconds": round(time.perf_counter() - t0, 1)}
    (out / "train_log.json").write_text(json.dumps(log, indent=2))

    m = best["metrics"]
    print(f"\nBest: lr={best['lr']:g} epoch {best['epoch']} | val acc {m['accuracy']:.3f}, "
          f"macro-F1 {m['macro_f1']:.3f}, expected cost {m['expected_cost']:.4f}")
    print("Val confusion (rows=true, cols=pred; LITE/MID/FULL):", m["confusion"])
    print(f"Saved model to {out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
