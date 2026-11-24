"""Split loading with a frozen-test guard (R7: test set untouched until the final report)."""
import hashlib
import json
import os
import time

import pandas as pd

from . import config as C


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_splits(splits: dict) -> None:
    C.DATA.mkdir(exist_ok=True)
    for name, rows in splits.items():
        with open(C.DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (C.DATA / "test.sha256").write_text(_sha(C.DATA / "test.jsonl"))


def verify_frozen() -> None:
    expected = (C.DATA / "test.sha256").read_text().strip()
    if _sha(C.DATA / "test.jsonl") != expected:
        raise RuntimeError("test.jsonl changed after it was frozen. Regenerate everything or restore it.")


def clusters(df: pd.DataFrame) -> pd.Series:
    """One cluster = one wording (topic x dialect x template). Messages in a cluster differ only in slot values,
    so confidence intervals resample clusters, not messages (DECISIONS.md D7)."""
    return df.topic + "|" + df.dialect + "|" + df.template.astype(str)


def load_split(name: str) -> pd.DataFrame:
    """train / val are free to use. test needs SHIFA_FINAL=1 and every access is logged."""
    if name == "test":
        if os.environ.get("SHIFA_FINAL") != "1":
            raise PermissionError("The test split is frozen. Run `make final` (sets SHIFA_FINAL=1).")
        verify_frozen()
        C.REP.mkdir(exist_ok=True)
        with open(C.REP / "test_access.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} test split opened by pid {os.getpid()}\n")
    return pd.read_json(C.DATA / f"{name}.jsonl", lines=True)
