"""Result record schema and JSON-lines storage.

One row = one measurement of one metric for one model on one test item.
Storing long-format rows keeps every pair compatible with the same
aggregation code, and the append-only file doubles as evidence of the run.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Record:
    pair: str  # e.g. "w2"
    model: str  # e.g. "qwen-image-edit-2509"
    item_id: str  # test item identifier, e.g. "room01_dusk"
    metric: str  # e.g. "clip_t", "lpips_bg", "latency_s", "peak_vram_gb"
    value: float
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


def append(path: str | Path, rec: Record) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def load(path: str | Path) -> list[Record]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[Record] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Record(**json.loads(line)))
    return out


def done_items(path: str | Path, model: str) -> set[str]:
    """Item ids already measured for a model; lets interrupted runs resume."""
    return {r.item_id for r in load(path) if r.model == model}
