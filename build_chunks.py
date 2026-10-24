"""Step 1: download 10-Ks from SEC and save clean Risk Factors chunks."""
import os
import pandas as pd
from src.sec_fetch import download_10k
from src.extract import html_to_text, extract_risk_section, split_into_chunks

TICKERS = ["AAPL", "JPM", "TSLA"]
YEARS = set(range(2019, 2026))


def main():
    rows = []
    for ticker in TICKERS:
        name, paths = download_10k(ticker, YEARS)
        for year, path in sorted(paths.items()):
            with open(path, encoding="utf-8", errors="ignore") as f:
                section = extract_risk_section(html_to_text(f.read()))
            for i, chunk in enumerate(split_into_chunks(section)):
                rows.append({"ticker": ticker, "company": name, "year": year, "chunk_id": i, "text": chunk})
            print(f"{ticker} {year}: {len(section.split()):,} words")
    os.makedirs("data/processed", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet("data/processed/chunks.parquet", index=False)
    print(f"Saved {len(df):,} chunks")


if __name__ == "__main__":
    main()
