"""Runtime router used by the app (router.md §8).

Loads the router once: tokenizer, temperature, thresholds, and the int8 ONNX model
(`model.int8.onnx`) if it has been exported. Until export_onnx.py exists the fp32
PyTorch checkpoint in the same folder is used instead; `backend` says which one is live.
Single thread, no network, deterministic.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.tiers import Tier

log = logging.getLogger(__name__)
ONNX_FILE = "model.int8.onnx"


@dataclass
class RouteDecision:
    tier: Tier               # final tier after the gate
    predicted_tier: Tier     # router's raw prediction
    confidence: float        # calibrated max probability, 0..1
    probs: list[float]       # calibrated probabilities [lite, mid, full]
    fallback: bool           # True if gate overrode the prediction
    tau: float               # threshold actually used
    latency_ms: float        # tokenize + infer + gate


class Router:
    def __init__(self, artifacts_dir: str | Path, max_length: int = 64, tau_override: float | None = None):
        from transformers import AutoTokenizer

        t0 = time.perf_counter()
        self.dir = Path(artifacts_dir)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir)
        self.temperature = json.loads((self.dir / "temperature.json").read_text())["temperature"]
        self.thresholds = json.loads((self.dir / "thresholds.json").read_text())
        self.tau_min = float(self.thresholds["tau_min"])
        self.default_tau = float(self.thresholds["default_tau"])
        # A manual τ replaces the risk-controlled one entirely (no clamping to tau_min).
        if tau_override is not None and not 0.0 < tau_override <= 1.0:
            raise ValueError(f"tau_override must be in (0, 1], got {tau_override}: confidence is a probability")
        self.tau_override = tau_override
        if tau_override is not None:
            log.warning("using manual tau_override=%.2f; the accuracy-loss bound from thresholds.json does not apply",
                        tau_override)

        onnx_path = self.dir / ONNX_FILE
        if onnx_path.exists():
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
            self.input_names = {i.name for i in self.session.get_inputs()}
            self.backend = "onnx-int8"
        else:
            import torch
            from transformers import AutoModelForSequenceClassification
            torch.set_num_threads(1)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.dir).eval()
            self.backend = "torch-fp32"
            log.warning("%s not found in %s; using the fp32 PyTorch checkpoint", ONNX_FILE, self.dir)
        self._logits("warm-up")  # first call pays one-time init cost; keep it out of route latency
        self.load_ms = (time.perf_counter() - t0) * 1000

    def clamp_tau(self, tau: float | None) -> float:
        if self.tau_override is not None:
            return self.tau_override
        tau = self.default_tau if tau is None else tau
        return min(max(tau, self.tau_min), 1.0)

    def _logits(self, text: str) -> np.ndarray:
        if self.backend == "onnx-int8":
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="np")
            feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            return self.session.run(None, feeds)[0][0]
        import torch
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            return self.model(**enc).logits[0].float().numpy()

    def route(self, text: str, tau: float | None = None) -> RouteDecision:
        t0 = time.perf_counter()
        tau = self.clamp_tau(tau)
        if not text or not text.strip():
            probs = [0.0, 0.0, 1.0]
            return RouteDecision(Tier.FULL, Tier.FULL, 0.0, probs, True, tau,
                                 (time.perf_counter() - t0) * 1000)
        z = self._logits(text) / self.temperature
        e = np.exp(z - z.max())
        probs = e / e.sum()
        predicted = Tier(int(probs.argmax()))
        confidence = float(probs.max())
        fallback = confidence < tau
        tier = Tier.FULL if fallback else predicted
        return RouteDecision(tier, predicted, confidence, [float(p) for p in probs], fallback, tau,
                             (time.perf_counter() - t0) * 1000)
