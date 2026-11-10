"""Keyword themes: simple, transparent counts that need no model."""
from __future__ import annotations

import re

import pandas as pd

# "AI" must be upper case so that it does not match ordinary words.
THEMES: dict[str, re.Pattern] = {
    "Artificial intelligence": re.compile(r"\bAI\b|[Aa]rtificial [Ii]ntelligence|[Mm]achine [Ll]earning"),
    "Pandemic / COVID-19": re.compile(r"covid|pandemic|coronavirus", re.I),
    "Tariffs & trade conflict": re.compile(r"\btariffs?\b|trade war|trade dispute|trade tension|export control", re.I),
    "Inflation": re.compile(r"\binflation", re.I),
    "Interest rates": re.compile(r"\binterest rates?\b", re.I),
    "Climate change": re.compile(r"climate change|greenhouse gas|carbon emission|net[- ]zero", re.I),
    "Cyber attacks": re.compile(r"cyber|ransomware|data breach|hack", re.I),
    "War & armed conflict": re.compile(r"\bwars?\b|ukraine|armed conflict|military conflict|hostilities", re.I),
    "Supply shortages": re.compile(r"shortage", re.I),
}


def custom_pattern(query: str) -> re.Pattern:
    """A user's search words as a safe, case-insensitive whole-word pattern."""
    return re.compile(r"\b" + re.escape(query.strip()), re.I)


def theme_share(passages: pd.DataFrame, pattern: re.Pattern) -> pd.DataFrame:
    """Per company and year: share of passages that mention the theme."""
    hit = passages["text"].str.contains(pattern)
    table = (
        passages.assign(hit=hit)
        .groupby(["company", "sector", "year"], as_index=False)
        .agg(passages=("hit", "size"), mentions=("hit", "sum"))
    )
    table["share"] = table["mentions"] / table["passages"]
    return table
