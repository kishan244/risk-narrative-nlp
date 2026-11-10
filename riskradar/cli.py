"""Command line: python -m riskradar <command>."""
from __future__ import annotations

import argparse
import json
import logging

from riskradar import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="riskradar", description="Risk language in SEC 10-K filings.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="download filings, classify every passage, write data/app/")
    sub.add_parser("train", help="train the classifier on data/eval/labels.csv and score it on the test set")
    analyze = sub.add_parser("analyze", help="run the pipeline on one company's 10-K")
    analyze.add_argument("ticker")
    analyze.add_argument("--year", type=int, help="fiscal year (default: newest filing)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "build":
        from riskradar.build import build

        info = build()
        print(json.dumps({k: info[k] for k in ("companies", "reports", "passages", "complete_years", "problems")}, indent=2))
    elif args.command == "train":
        from riskradar.train import train

        metrics = train()
        print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion", "per_category")}, indent=2))
    else:
        from riskradar.analyze import analyze_filing
        from riskradar.classify import RiskClassifier
        from riskradar.embed import Embedder
        from riskradar.sec import SecClient

        result = analyze_filing(
            args.ticker, SecClient(), Embedder(), RiskClassifier.load(), args.year,
            on_step=lambda name, seconds, detail: print(f"{name:32s} {seconds:5.1f}s  {detail}"),
        )
        print()
        for category, value in result.mix.sort_values(ascending=False).items():
            print(f"{value:6.1%}  {category}")
    return 0
