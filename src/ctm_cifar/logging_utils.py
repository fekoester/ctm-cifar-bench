from __future__ import annotations

import os
import json
import csv
import time
from typing import Any, Dict, List, Optional


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def now_unix() -> float:
    return time.time()


def write_json(path: str, obj: Any):
    safe_makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_jsonl(path: str, row: Dict[str, Any]):
    safe_makedirs(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


class CSVLogger:
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        safe_makedirs(os.path.dirname(path))
        exists = os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        if not exists:
            self.w.writeheader()
            self.f.flush()

    def log(self, row: Dict[str, Any]):
        clean = {k: row.get(k, "") for k in self.fieldnames}
        self.w.writerow(clean)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

