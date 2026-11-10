"""Paths and the company list."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"      # not in git: downloaded sections, embeddings
APP_DIR = ROOT / "data" / "app"          # small files the Streamlit app reads
EVAL_DIR = ROOT / "data" / "eval"        # labelled passages (train + test)
MODEL_DIR = ROOT / "models"              # trained classifier weights + metrics


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    sector: str
    cik: str | None = None  # only needed when the ticker now points to a new legal entity


def load_companies(path: Path | None = None) -> tuple[int, list[Company]]:
    """Return (start_year, companies) from companies.toml."""
    with open(path or ROOT / "companies.toml", "rb") as f:
        raw = tomllib.load(f)
    return raw["start_year"], [Company(**c) for c in raw["company"]]
