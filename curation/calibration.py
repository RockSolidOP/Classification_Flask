from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple, Optional


@dataclass
class PlattModel:
    A: float
    B: float
    n: int
    pos: int
    neg: int
    created_at: str

    def predict(self, score: float) -> float:
        # Logistic: p = 1 / (1 + exp(A * s + B))
        import math
        z = self.A * float(score) + self.B
        # numerically stable sigmoid
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)


def _sigmoid(x):
    import math
    if x >= 0:
        ez = math.exp(-x)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)


def fit_platt(scores: Iterable[float], labels: Iterable[int], lr: float = 0.1, l2: float = 1e-4, epochs: int = 1500) -> PlattModel:
    xs = [float(s) for s in scores]
    ys = [1 if int(y) == 1 else 0 for y in labels]
    n = len(xs)
    pos = sum(ys)
    neg = n - pos
    if n == 0 or pos == 0 or neg == 0:
        # Degenerate; return near-uniform
        return PlattModel(A=0.0, B=0.0, n=n, pos=pos, neg=neg, created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z")

    # Initialize A, B (bias B to empirical logit)
    import math
    # Guard against extreme class imbalance; clamp prior to (eps, 1-eps)
    prior = (pos + 1.0) / (neg + 1.0)
    eps = 1e-6
    if prior <= 0.0:
        prior = eps
    if prior >= 1.0:
        prior = 1.0 - eps
    # Initialize with a bias toward the empirical prior
    A, B = 0.0, -math.log(1.0 / prior - 1.0)

    for _ in range(epochs):
        gA = 0.0
        gB = 0.0
        hAA = l2
        hBB = l2
        hAB = 0.0
        # Accumulate gradients and Hessian (IRLS step)
        for s, y in zip(xs, ys):
            z = A * s + B
            p = _sigmoid(z)
            w = p * (1.0 - p)  # sigmoid derivative
            g = p - y
            gA += g * s
            gB += g
            hAA += w * s * s
            hBB += w
            hAB += w * s
        # Solve 2x2 system for Newton step: H * step = -g
        det = hAA * hBB - hAB * hAB
        if det <= 0:
            # fallback to small gradient step
            A -= lr * gA
            B -= lr * gB
        else:
            dA = (-gA * hBB + gB * hAB) / det
            dB = (-gB * hAA + gA * hAB) / det
            A += dA
            B += dB
        # Optional early stop if small updates
        if abs(gA) < 1e-6 and abs(gB) < 1e-6:
            break

    return PlattModel(A=A, B=B, n=n, pos=pos, neg=neg, created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z")


def save_platt(model: PlattModel, path: Path, note: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "method": "platt",
        "A": model.A,
        "B": model.B,
        "n": model.n,
        "pos": model.pos,
        "neg": model.neg,
        "created_at": model.created_at,
    }
    if note:
        d["note"] = note
    with path.open("w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_platt(path: Path) -> Optional[PlattModel]:
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("method") != "platt":
            return None
        return PlattModel(A=float(d["A"]), B=float(d["B"]), n=int(d.get("n", 0)), pos=int(d.get("pos", 0)), neg=int(d.get("neg", 0)), created_at=str(d.get("created_at", "")))
    except Exception:
        return None
