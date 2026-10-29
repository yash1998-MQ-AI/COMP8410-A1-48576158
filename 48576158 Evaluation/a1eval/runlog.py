"""Run logging: latency, peak GPU memory and the GPU cost ledger."""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .schema import Record, append

COST_HEADER = ["date", "provider", "gpu", "hours", "rate_usd_per_hr", "cost_usd", "note"]


def _cuda():
    try:
        import torch

        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


@contextmanager
def timed(
    results_path: str | Path,
    pair: str,
    model: str,
    item_id: str,
    seed: int | None = None,
    **meta: Any,
) -> Iterator[dict[str, Any]]:
    """Time a generation call and log latency_s and peak_vram_gb.

    Usage:
        with timed("results/w2.jsonl", "w2", "qwen", "room01", seed=0) as ctx:
            image = pipe(...)
            ctx["prompt"] = prompt   # extra fields end up in meta
    """
    torch = _cuda()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    ctx: dict[str, Any] = dict(meta)
    start = time.perf_counter()
    yield ctx
    if torch is not None:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    append(results_path, Record(pair, model, item_id, "latency_s", elapsed, seed, ctx))
    if torch is not None:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        append(results_path, Record(pair, model, item_id, "peak_vram_gb", peak, seed, ctx))


def log_cost(
    path: str | Path,
    date: str,
    provider: str,
    gpu: str,
    hours: float,
    rate_usd_per_hr: float,
    note: str = "",
) -> None:
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COST_HEADER)
        w.writerow([date, provider, gpu, hours, rate_usd_per_hr, round(hours * rate_usd_per_hr, 4), note])


def total_cost(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(float(r["cost_usd"]) for r in csv.DictReader(fh))
