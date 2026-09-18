#!/usr/bin/env python3
"""
Reshape flat tactile vectors (1280,) into (5, 16, 16) for LeRobot datasets.
Also updates meta/info.json and prints a validation summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_T = 5
DEFAULT_H = 16
DEFAULT_W = 16
DEFAULT_FLAT = DEFAULT_T * DEFAULT_H * DEFAULT_W


def _iter_parquet_files(dataset_dir: Path) -> Iterable[Path]:
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        yield parquet_path


def _reshape_fsr_array(x: np.ndarray, *, t: int, h: int, w: int) -> list:
    x = np.asarray(x, dtype=np.float32)
    if x.shape == (t, h, w):
        return x.tolist()
    if x.shape == (t * h * w,):
        return x.reshape(t, h, w).tolist()
    raise ValueError(f"Unexpected tactile shape: {x.shape}")


def _to_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype != object:
        return arr

    def _stack(obj):
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            return np.stack([_stack(v) for v in obj], axis=0)
        return np.asarray(obj)

    return _stack(arr)


def _process_parquet(parquet_path: Path, *, t: int, h: int, w: int, write: bool) -> dict:
    df = pd.read_parquet(parquet_path)
    required = ["observation.tactile_left", "observation.tactile_right"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {parquet_path}")

    for col in required:
        df[col] = df[col].apply(lambda x: _reshape_fsr_array(x, t=t, h=h, w=w))

    if write:
        df.to_parquet(parquet_path, index=False)

    return {"rows": len(df)}


def _update_info_json(dataset_dir: Path, *, t: int, h: int, w: int, write: bool) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing meta/info.json: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    for key in ("observation.tactile_left", "observation.tactile_right"):
        if key not in info.get("features", {}):
            continue
        info["features"][key]["shape"] = [t, h, w]
        info["features"][key]["dtype"] = "float32"
        info["features"][key]["description"] = f"FSR window [k={t}, {h}x{w}]"

    if write:
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_dataset(dataset_dir: Path, *, t: int, h: int, w: int) -> dict:
    summary = {
        "parquet_files": 0,
        "total_rows": 0,
        "bad_rows": 0,
    }
    for parquet_path in _iter_parquet_files(dataset_dir):
        df = pd.read_parquet(parquet_path)
        summary["parquet_files"] += 1
        summary["total_rows"] += len(df)
        for col in ("observation.tactile_left", "observation.tactile_right"):
            bad = df[col].apply(lambda x: _to_array(x).shape != (t, h, w)).sum()
            summary["bad_rows"] += int(bad)

    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info_shape = info.get("features", {}).get("observation.tactile_left", {}).get("shape")
    summary["info_shape"] = info_shape
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/hz425/openpi/openpi-VTLA/uf850_teleop_dataset0224"),
    )
    parser.add_argument("--t", type=int, default=DEFAULT_T)
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--w", type=int, default=DEFAULT_W)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not args.check_only:
        for parquet_path in _iter_parquet_files(args.dataset):
            _process_parquet(parquet_path, t=args.t, h=args.h, w=args.w, write=True)
        _update_info_json(args.dataset, t=args.t, h=args.h, w=args.w, write=True)

    summary = _validate_dataset(args.dataset, t=args.t, h=args.h, w=args.w)
    print("Validation summary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")

    if summary["bad_rows"] == 0 and summary["info_shape"] == [args.t, args.h, args.w]:
        print("Dataset looks compatible with tactile training requirements.")
    else:
        print("Dataset still has issues. Please review the summary.")


if __name__ == "__main__":
    main()
