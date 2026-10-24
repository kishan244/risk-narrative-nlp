"""Risk Radar: how risk language in 10-K annual reports changes over time.

Reads only precomputed files from data/app/, so it loads fast and needs no
machine-learning libraries at runtime.
"""
import json

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Risk Radar · 10-K risk language", page_icon="📈", layout="wide")


@st.cache_data
def load():
    chunks = pd.read_parquet("data/app/chunks.parquet")
    share = pd.read_csv("data/app/category_share.csv")
    new = pd.read_csv("data/app/new_language.csv")
    with open("data/app/metrics.json") as f:
        metrics = json.load(f)
    return chunks, share, new, metrics


chunks, share, new, metrics = load()
ALL_CATS = sorted(share["category"].unique())
COLORS = dict(zip(ALL_CATS, px.colors.qualitative.Dark24))
companies = sorted(chunks["company"].unique())
years = sorted(chunks["year"].unique())

st.title("Risk Radar")
st.markdown(
    "How do **Apple, JPMorgan and Tesla** describe their risks, and how did that change from "
    f"**{years[0]} to {years[-1]}**? Every chart comes from the *Item 1A Risk Factors* section of "
    "their official 10-K annual reports, downloaded from SEC EDGAR and read by an NLP model."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Annual reports", metrics["reports"])
k2.metric("Text chunks analysed", f"{metrics['chunks']:,}")
k3.metric("Risk categories", metrics["categories"])
k4.metric("Accuracy (cross-validated)", f"{metrics['test_accuracy']:.0%}")

tab_trend, tab_compare, tab_change, tab_read, tab_method = st.tabs(
    ["Risk mix over time", "Compare companies", "What changed", "Read the text", "Method & limits"]
)

with tab_trend:
    company = st.selectbox("Company", companies, key="trend_company")
    d = share[share["company"] == company]
    fig = px.area(
        d, x="year", y="share", color="category",
        labels={"share": "Share of risk section", "year": ""},
        category_orders={"category": ALL_CATS}, color_discrete_map=COLORS,
    )
    fig.update_layout(yaxis_tickformat=".0%", legend_title_text="", height=480, margin=dict(t=10))
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Share = chunks in a category ÷ all chunks in that report. Using shares, not raw counts, "
        "means a longer report does not look riskier just because it is longer."
    )

with tab_compare:
    year = st.select_slider("Year", options=years, value=years[-1])
    d = share[share["year"] == year].pivot(index="category", columns="company", values="share").fillna(0)
    fig = px.imshow(
        d, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
        labels={"color": "Share"},
    )
    fig.update_layout(height=520, margin=dict(t=10), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Each column adds up to 100%. Darker = a bigger part of that company's risk section.")

with tab_change:
    company = st.selectbox("Company", companies, key="change_company")
    d = share[share["company"] == company].pivot(index="category", columns="year", values="share").fillna(0)
    first, last = d.columns.min(), d.columns.max()
    delta = (d[last] - d[first]).sort_values()
    delta_df = delta.reset_index().rename(columns={0: "change"})
    delta_df["direction"] = delta_df["change"].apply(lambda v: "More attention" if v > 0 else "Less attention")
    fig = px.bar(
        delta_df, x="change", y="category", orientation="h", color="direction",
        color_discrete_map={"More attention": "#d62728", "Less attention": "#2ca02c"},
        labels={"change": f"Change in share, {first} → {last} (percentage points)", "category": ""},
    )
    fig.update_layout(xaxis_tickformat="+.0%", height=460, margin=dict(t=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"New risk language in {last}")
    st.caption(
        f"Passages from {company}'s {last} report that are least similar to anything in its "
        f"{first} report: a simple way to spot risks that were not talked about before."
    )
    for _, row in new[new["company"] == company].iterrows():
        with st.expander(f"{row['category']} · novelty {row['novelty']:.2f}"):
            st.write(row["text"])

with tab_read:
    c1, c2, c3 = st.columns(3)
    company = c1.selectbox("Company", companies, key="read_company")
    year = c2.selectbox("Year", years, index=len(years) - 1, key="read_year")
    cats = sorted(chunks["category"].unique())
    cat = c3.selectbox("Category", ["All"] + cats, key="read_cat")
    d = chunks[(chunks["company"] == company) & (chunks["year"] == year)]
    if cat != "All":
        d = d[d["category"] == cat]
    st.caption(f"{len(d)} chunks")
    for _, row in d.head(40).iterrows():
        with st.expander(f"#{row['chunk_id']} · {row['category']}"):
            st.write(row["text"])

with tab_method:
    st.markdown(
        f"""
**Pipeline**
1. **Download**: 10-K filings for AAPL, JPM and TSLA, fiscal years {years[0]}–{years[-1]}, straight from SEC EDGAR.
2. **Extract**: find the real *Item 1A Risk Factors* section (not the table of contents) and cut it at *Item 1B / 1C / 2*.
3. **Chunk**: split into ~200-word passages along paragraph and sentence borders.
4. **Classify**: embed each passage with `{metrics['model']}`, score it against short example sentences for {metrics['categories']} risk categories,
   then let a logistic-regression model trained on hand-labelled passages pick the category. The same model judges a bank, a phone maker and a car maker.
5. **Measure**: share of each category per report, change over time, and passages with new wording.

**How good is it?**
- **{metrics['test_n']} random passages** were labelled by hand with help from an AI assistant (Claude); half were labelled only after the method was frozen.\n  All labels are public in `data/eval/` so anyone can check them.
- A small logistic-regression model on top of the embeddings, trained on those labels, gets
  **{metrics['test_accuracy']:.0%}** right in 10-fold cross-validation. Always guessing the most common category gets {metrics['test_baseline']:.0%}.
- Repeated paragraphs (the same text in several years) are kept in the same fold, so the model is never tested on text it has already seen.
  Without that rule the score looked higher, but that number was not honest.
- The first version (example sentences only, no training) scored {metrics['zero_shot_test_accuracy']:.0%} on the fresh half.

**Limits**
- Many passages mention two risks at once (e.g. litigation *and* reputation), but each gets one label.
- Three companies only, so this shows how the method works, not an industry-wide finding.
- Not investment advice.
"""
    )
    st.markdown("Code: [github.com/kishan244/risk-narrative-nlp](https://github.com/kishan244/risk-narrative-nlp)")
