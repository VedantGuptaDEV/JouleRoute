"""Helpers shared by the training, calibration and threshold scripts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.config import repo_path  # noqa: E402
from shared.tiers import Tier  # noqa: E402

N_TIERS = len(Tier)
BASELINE = "baseline"


def artifacts_dir(cfg, name: str) -> Path:
    return repo_path(cfg["router"]["artifacts_root"]) / name


def results_dir(cfg) -> Path:
    path = repo_path(cfg["router"]["results_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_labeled(cfg, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(repo_path(cfg["router"]["data_dir"]) / "labeled.csv")
    return df if split is None else df[df.split == split].reset_index(drop=True)


def cost_matrix(cfg) -> np.ndarray:
    return np.asarray(cfg["router"]["cost_matrix"], dtype=np.float64)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_cost(probs: np.ndarray, labels: np.ndarray, C: np.ndarray) -> float:
    """Mean over queries of sum_j p_j * C[y, j]."""
    return float((probs * C[labels]).sum(axis=1).mean())


def classification_metrics(probs: np.ndarray, labels: np.ndarray, C: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", labels=list(range(N_TIERS)), zero_division=0)),
        "expected_cost": expected_cost(probs, labels, C),
        "confusion": confusion_matrix(labels, pred, labels=list(range(N_TIERS))).tolist(),
    }


def load_temperature(cfg, name: str) -> float:
    path = artifacts_dir(cfg, name) / "temperature.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run calibrate.py --name {name} first")
    return json.loads(path.read_text())["temperature"]


def get_logits(cfg, name: str, texts: list[str]) -> np.ndarray:
    """Uncalibrated logits [n, 3] for a trained router ('baseline' or a fine-tuned HF model)."""
    if name == BASELINE:
        import joblib
        model = joblib.load(artifacts_dir(cfg, name) / "baseline.joblib")
        return np.log(np.clip(model.predict_proba(texts), 1e-12, None))
    return hf_logits(artifacts_dir(cfg, name), texts, cfg["router"]["max_length"])


def hf_logits(model_dir: Path, texts: list[str], max_length: int, batch_size: int = 256) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt")
            out.append(model(**enc).logits.float().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_TIERS))
