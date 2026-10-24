# Risk Radar: how companies' risk language changes over time

**Live demo:** https://risk-radar-kishan-v1.streamlit.app/

An end-to-end NLP pipeline that downloads 10-K annual reports from **SEC EDGAR**, extracts the *Item 1A Risk Factors* section, sorts every passage into one of **12 risk categories**, and shows how each company's risk mix changed from **2019 to 2025** in a Streamlit app.

Companies: **Apple, JPMorgan Chase, Tesla** (a phone maker, a bank and a car maker, so the method has to work across very different industries).

## Key results

| | |
|---|---|
| Reports analysed | 21 official 10-K filings (3 companies × 7 fiscal years, 2019–2025) |
| Passages | 1,287 (~200 words each, split on paragraph and sentence borders) |
| Classifier accuracy | **69%** in 10-fold cross-validation on 118 hand-labelled passages (most-common-class baseline: 25%) |
| Apple | Regulatory & legal risk grew the most: **17% → 32%** of the risk section |
| Tesla | Product & technology risk up **+10 points** (FSD, robotaxi, Optimus); supply-chain risk down **−7 points** |
| JPMorgan | Cybersecurity up **+6 points**; 2025 report adds new language on generative and agentic AI |

Shares, not raw counts, are used everywhere, so a longer report does not look "riskier" just because it is longer.

## Pipeline

```
SEC EDGAR ──► build_chunks.py ──► embed_all.py ──► build_results.py ──► app.py (Streamlit)
  10-K HTML     Item 1A text,       bge-base          classify, shares,     5 interactive views
                ~200-word chunks    embeddings        new language, metrics
```

1. **Download** (`src/sec_fetch.py`): ticker → CIK → every 10-K filing via the SEC submissions API. Cached, rate-limited, with the User-Agent SEC requires.
2. **Extract** (`src/extract.py`): the table of contents also says "Item 1A Risk Factors", so the code takes the *longest* section that runs from an Item 1A heading to the next Item 1B / 1C / 2 heading (or the next Item 1A heading, so cross-references don't swallow other sections). Covered by unit tests.
3. **Chunk**: paragraph-aware ~200-word passages; very long paragraphs are split at sentence ends.
4. **Embed** (`embed_all.py`): `BAAI/bge-base-en-v1.5` sentence embeddings, cached so re-runs only embed changed passages.
5. **Classify** (`build_results.py`): each passage is scored against short example sentences per category (`src/prototypes.py`), then a logistic-regression model trained on the hand-labelled passages picks the category.
6. **Measure**: category share per report, change 2019 → 2025, and "new language" (2025 passages least similar to anything in the 2019 report).

## How the accuracy was measured (and why it is not higher)

- 118 random passages were labelled by hand with help from an AI assistant (Claude). All labels are in `data/eval/` so anyone can check them.
- **First attempt, zero-shot** (example sentences only): 80% on the passages used while writing the examples, but only **58% on a fresh sample**. The examples had been tuned to the first sample, so I froze the method and kept the fresh sample as an honest test.
- **Final model**: logistic regression on embeddings + example-sentence scores, **69%** in 10-fold cross-validation.
- 10-K filings repeat the same paragraph year after year. Plain cross-validation therefore tested the model on text it had already seen and reported 84%. Near-identical passages (cosine ≥ 0.9) are now kept in the same fold, which gives the honest 69%.

## Limits

- Many passages mention two risks at once; each gets one label.
- Three companies only: this shows the method, not an industry-wide finding.
- 118 labels is a small test set, so the accuracy has a margin of roughly ±8 points.
- Not investment advice.

## Run it yourself

```bash
pip install -r requirements-pipeline.txt
python build_chunks.py      # downloads 21 filings from SEC (~150 MB, cached in data/sec/)
python embed_all.py         # ~15 min on a laptop CPU, cached afterwards
python build_results.py     # writes data/app/
streamlit run app.py
pytest
```

The live app only needs `requirements.txt` (Streamlit, pandas, Plotly) because it reads the precomputed files in `data/app/`.

## Project history

The first version (semester project) read PDFs and used BERTopic. A review showed its topics followed the *company*, not the *risk type* (one topic held 412 JPMorgan passages and 0 from Apple), and that JPMorgan's "risk" text actually came from the financial statements. It is kept in `legacy/` for reference.

## Tech stack

Python · requests · BeautifulSoup · sentence-transformers (bge-base) · scikit-learn · pandas · Plotly · Streamlit · pytest
