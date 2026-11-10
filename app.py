"""Risk Radar: how companies describe their risks in SEC 10-K filings.

The first four tabs read small precomputed files from data/app/, so they load
fast. The "Live analysis" tab runs the whole pipeline on demand for any ticker.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from riskradar.config import APP_DIR
from riskradar.taxonomy import CATEGORIES, NON_RISK, RISK_CATEGORIES
from riskradar.themes import THEMES, custom_pattern, theme_share

st.set_page_config(page_title="Risk Radar · 10-K risk language", page_icon="📡", layout="wide")

# ---------------------------------------------------------------- look and feel
BLUE, RED, GREY, INK, MUTED = "#2a78d6", "#e34948", "#b9bcc2", "#0b0b0b", "#52514e"
SEQUENTIAL = [[0, "#f6f8fb"], [0.15, "#cde2fb"], [0.4, "#6da7ec"], [0.7, "#256abf"], [1, "#0d366b"]]
SECTOR_COLORS = {
    "Technology": "#2a78d6", "Consumer": "#eb6834", "Autos & Industrials": "#1baf7a",
    "Financials": "#eda100", "Health Care": "#e87ba4", "Energy": "#008300",
}
PLOT_CONFIG = {"displayModeBar": False}


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), font=dict(color=INK, size=13),
        plot_bgcolor="white", paper_bgcolor="white", legend_title_text="", hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor="#eceef1", zeroline=False, title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#eceef1", zeroline=False, title_font=dict(color=MUTED))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def heatmap(table: pd.DataFrame, height: int) -> go.Figure:
    """Shares as a one-colour heatmap: light = little text, dark = a lot."""
    fig = px.imshow(
        table, text_auto=".0%", aspect="auto", color_continuous_scale=SEQUENTIAL,
        zmin=0, zmax=max(0.3, float(np.nanmax(table.to_numpy()))),
    )
    fig.update_traces(hovertemplate="%{y}<br>%{x}: <b>%{z:.1%}</b><extra></extra>", xgap=2, ygap=2)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(side="top", title=None, tickangle=0, type="category")
    fig.update_yaxes(title=None)
    return style(fig, height)


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    passages = pd.read_parquet(APP_DIR / "passages.parquet")
    share = pd.read_csv(APP_DIR / "category_share.csv")
    reports = pd.read_csv(APP_DIR / "reports.csv")
    info = json.loads((APP_DIR / "build_info.json").read_text())
    return passages, share, reports, info


@st.cache_data
def load_embeddings() -> np.ndarray:
    return np.load(APP_DIR / "embeddings.npy").astype(np.float32)


@st.cache_resource(show_spinner="Loading the embedding model (first use only)…")
def get_embedder():
    from riskradar.embed import Embedder

    embedder = Embedder()
    embedder.encode(["warm up"])
    return embedder


@st.cache_resource
def get_classifier():
    from riskradar.classify import RiskClassifier

    return RiskClassifier.load()


@st.cache_resource
def get_sec_client():
    from riskradar.sec import SecClient

    return SecClient()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_theme(pattern_text: str, flags: int) -> pd.DataFrame:
    passages, *_ = load_data()
    return theme_share(passages, re.compile(pattern_text, flags))


passages, share, reports, info = load_data()
metrics = info["metrics"]
YEARS = info["complete_years"]          # years in which (nearly) every company has filed
FIRST, LAST = YEARS[0], YEARS[-1]
COMPANIES = sorted(reports["company"].unique())
SECTORS = list(SECTOR_COLORS)
benchmark = share[share["year"] == LAST].groupby("category")["share"].mean()

# ---------------------------------------------------------------- header
st.title("Risk Radar")
st.markdown(
    f"What do big US companies say could go wrong, and how is that changing? "
    f"This app reads the **Risk Factors** section of **{info['reports']} annual reports (10-K)** from "
    f"**{info['companies']} companies**, straight from the SEC, and sorts every passage into "
    f"**{len(RISK_CATEGORIES)} risk types** with an NLP model."
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Companies", info["companies"], help="Six sectors: " + ", ".join(SECTORS))
k2.metric("Annual reports", info["reports"], help=f"Fiscal years {info['first_year']}–{info['last_year']}")
k3.metric("Passages classified", f"{info['passages']:,}")
k4.metric(
    "Accuracy on unseen test set", f"{metrics['test_accuracy']:.1%}",
    help=f"{metrics['test_passages']} labelled passages the model never saw. "
         f"Guessing the most common class would score {metrics['baseline_most_common_class']:.0%}.",
)

tab_overview, tab_company, tab_compare, tab_themes, tab_live, tab_method = st.tabs(
    ["Overview", "Company", "Compare", "Themes", "⚡ Live analysis", "Method & accuracy"]
)

# ================================================================ OVERVIEW
with tab_overview:
    st.subheader("Three things the data shows")
    for column, insight in zip(st.columns(3), info["insights"]):
        with column.container(border=True):
            st.caption(insight["title"].upper())
            st.markdown(f"### {insight['value']}")
            st.write(insight["text"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Talk about AI took off in 2023**")
        st.caption("Share of risk passages that mention AI, average of all companies")
        ai = cached_theme(THEMES["Artificial intelligence"].pattern, THEMES["Artificial intelligence"].flags)
        ai = ai[ai["year"].isin(YEARS)].groupby("year", as_index=False)["share"].mean()
        fig = px.line(ai, x="year", y="share", markers=True, color_discrete_sequence=[BLUE])
        fig.update_traces(line_width=2.5, marker_size=9, hovertemplate="%{x}: <b>%{y:.1%}</b><extra></extra>")
        fig.add_annotation(
            x=ai["year"].iloc[-1], y=ai["share"].iloc[-1], text=f"<b>{ai['share'].iloc[-1]:.0%}</b>",
            showarrow=False, xshift=-4, yshift=16, font=dict(color=INK),
        )
        fig.update_yaxes(tickformat=".0%", title=None, rangemode="tozero")
        fig.update_xaxes(dtick=1, title=None)
        show(style(fig, 300))
    with right:
        st.markdown("**Most risk text is copied from the year before**")
        st.caption("Share of sentences repeated word for word from last year's report, average of all companies")
        reuse = reports[reports["year"].isin(YEARS)].dropna(subset=["reused_share"])
        reuse = reuse.groupby("year", as_index=False)["reused_share"].mean()
        fig = px.bar(reuse, x="year", y="reused_share", color_discrete_sequence=[BLUE], text_auto=".0%")
        fig.update_traces(hovertemplate="%{x}: <b>%{y:.1%}</b> repeated<extra></extra>", marker_cornerradius=4, width=0.6)
        fig.update_yaxes(tickformat=".0%", title=None, range=[0, 1])
        fig.update_xaxes(dtick=1, title=None)
        show(style(fig, 300))

    st.markdown(f"**Each sector worries about different things** · {LAST}")
    st.caption("Share of the risk section per risk type, average of the companies in each sector. Columns add up to 100%.")
    sector = share[share["year"] == LAST].groupby(["category", "sector"])["share"].mean().unstack()[SECTORS]
    sector = sector.loc[sector.mean(axis=1).sort_values(ascending=False).index]
    show(heatmap(sector, 470))

# ================================================================ COMPANY
with tab_company:
    company = st.selectbox("Company", COMPANIES, index=COMPANIES.index("Apple"))
    own = share[share["company"] == company]
    own_reports = reports[reports["company"] == company].sort_values("year")
    newest, oldest = own_reports.iloc[-1], own_reports.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sector", newest["sector"])
    c2.metric("Newest report", f"FY {newest['year']}", help=f"Filed {newest['filed']}")
    grown = newest["words"] / oldest["words"] - 1
    c3.metric("Words in risk section", f"{newest['words']:,}", f"{grown:+.0%} since {oldest['year']}", delta_color="off", delta_arrow="off")
    if pd.notna(newest["reused_share"]):
        c4.metric("Copied from last year", f"{newest['reused_share']:.0%}", help="Sentences repeated word for word")
    st.caption(f"Source: [{company} 10-K for fiscal year {newest['year']} on SEC EDGAR]({newest['url']})")

    st.markdown("**Risk mix by year**")
    st.caption("Share of the risk section per risk type. Each column adds up to 100%.")
    table = own.pivot(index="category", columns="year", values="share")
    table = table.loc[table[table.columns[-1]].sort_values(ascending=False).index]
    table.columns = [str(c) for c in table.columns]
    show(heatmap(table, 470))

    first_year, last_year = int(own["year"].min()), int(own["year"].max())
    delta = (table[str(last_year)] - table[str(first_year)]).sort_values()
    delta = delta[delta.abs() >= 0.005]
    left, right = st.columns([5, 6])
    with left:
        st.markdown(f"**What changed, {first_year} → {last_year}**")
        st.caption("Change in share, in percentage points")
        colors = [RED if v > 0 else BLUE for v in delta]
        fig = go.Figure(go.Bar(
            x=delta.values * 100, y=delta.index, orientation="h", marker_color=colors, marker_cornerradius=4,
            text=[f"{v * 100:+.1f}" for v in delta], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: <b>%{x:+.1f} points</b><extra></extra>",
        ))
        pad = max(abs(delta.min()), abs(delta.max())) * 100 * 0.25 + 0.5
        fig.update_xaxes(title=None, zeroline=True, zerolinecolor=GREY, showticklabels=False,
                         range=[min(delta.min() * 100, 0) - pad, max(delta.max() * 100, 0) + pad])
        show(style(fig, max(260, 30 * len(delta) + 40)))
        st.caption("🟥 more attention · 🟦 less attention")
    with right:
        st.markdown(f"**Newest topics in the {last_year} report**")
        st.caption("Passages least similar to anything in the report of the year before")
        fresh = passages[(passages["company"] == company) & (passages["year"] == last_year)]
        fresh = fresh[~fresh["category"].isin(NON_RISK)].dropna(subset=["similarity_prev"])
        for _, row in fresh.nsmallest(4, "similarity_prev").iterrows():
            with st.expander(f"{row['category']} · {row['text'][:70]}…"):
                st.write(row["text"])

# ================================================================ COMPARE
with tab_compare:
    c1, c2 = st.columns([2, 1])
    category = c1.selectbox("Risk type", RISK_CATEGORIES, index=RISK_CATEGORIES.index("Cybersecurity & Data Privacy"))
    year = c2.select_slider("Year", options=sorted(share["year"].unique()), value=LAST)
    st.caption(CATEGORIES[category])

    left, right = st.columns([6, 5])
    with left:
        st.markdown(f"**Who writes most about it?** · {year}")
        ranking = share[(share["category"] == category) & (share["year"] == year)].sort_values("share")
        fig = px.bar(
            ranking, x="share", y="company", color="sector", orientation="h", text_auto=".0%",
            color_discrete_map=SECTOR_COLORS, category_orders={"sector": SECTORS, "company": ranking["company"].tolist()[::-1]},
        )
        fig.update_traces(hovertemplate="%{y}: <b>%{x:.1%}</b><extra></extra>", marker_cornerradius=4)
        fig.update_xaxes(tickformat=".0%", title="Share of the risk section")
        fig.update_yaxes(title=None)
        fig.update_layout(legend=dict(orientation="h", y=-0.12))
        show(style(fig, max(320, 24 * len(ranking) + 110)))
        if year not in YEARS:
            st.caption(f"Only {len(ranking)} companies have filed their {year} report so far.")
    with right:
        st.markdown("**Trend by sector**")
        trend = share[(share["category"] == category) & share["year"].isin(YEARS)]
        trend = trend.groupby(["year", "sector"], as_index=False)["share"].mean()
        fig = px.line(
            trend, x="year", y="share", color="sector", markers=True,
            color_discrete_map=SECTOR_COLORS, category_orders={"sector": SECTORS},
        )
        fig.update_traces(line_width=2, marker_size=7, hovertemplate="%{x}: <b>%{y:.1%}</b>")
        fig.update_yaxes(tickformat=".0%", title=None, rangemode="tozero")
        fig.update_xaxes(dtick=1, title=None)
        fig.update_layout(legend=dict(orientation="h", y=-0.15), hovermode="x unified")
        show(style(fig, 430))

    with st.expander("Table view"):
        wide = share[share["category"] == category].pivot(index="company", columns="year", values="share")
        st.dataframe(wide.style.format("{:.1%}", na_rep="–"), width="stretch")

# ================================================================ THEMES
with tab_themes:
    st.markdown("Track a word or topic through seven years of risk factors. This tab uses plain keyword counts, no model.")
    c1, c2 = st.columns([1, 1])
    choice = c1.selectbox("Theme", list(THEMES))
    query = c2.text_input("…or type your own word or phrase", placeholder="for example: semiconductor, strike, obesity")
    if query.strip():
        pattern, label = custom_pattern(query), f"“{query.strip()}”"
    else:
        pattern, label = THEMES[choice], choice
    theme = cached_theme(pattern.pattern, pattern.flags)
    theme = theme[theme["year"].isin(YEARS)]

    if theme["mentions"].sum() == 0:
        st.info(f"No passage mentions {label}. Try another word.")
    else:
        first_n = int((theme[theme["year"] == FIRST]["mentions"] > 0).sum())
        last_n = int((theme[theme["year"] == LAST]["mentions"] > 0).sum())
        peak = theme.groupby("year")["share"].mean()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Companies mentioning it in {FIRST}", f"{first_n} of {theme[theme['year'] == FIRST]['company'].nunique()}")
        m2.metric(f"Companies mentioning it in {LAST}", f"{last_n} of {theme[theme['year'] == LAST]['company'].nunique()}")
        m3.metric("Peak year", int(peak.idxmax()), f"{peak.max():.1%} of passages", delta_color="off", delta_arrow="off")

        left, right = st.columns(2)
        with left:
            st.markdown(f"**{label}: share of risk passages that mention it**")
            by_sector = st.toggle("Split by sector", value=False)
            if by_sector:
                data = theme.groupby(["year", "sector"], as_index=False)["share"].mean()
                fig = px.line(data, x="year", y="share", color="sector", markers=True,
                              color_discrete_map=SECTOR_COLORS, category_orders={"sector": SECTORS})
                fig.update_layout(legend=dict(orientation="h", y=-0.15), hovermode="x unified")
                fig.update_traces(line_width=2, marker_size=7, hovertemplate="%{x}: <b>%{y:.1%}</b>")
            else:
                data = theme.groupby("year", as_index=False)["share"].mean()
                fig = px.line(data, x="year", y="share", markers=True, color_discrete_sequence=[BLUE])
                fig.update_traces(line_width=2.5, marker_size=9, hovertemplate="%{x}: <b>%{y:.1%}</b><extra></extra>")
            fig.update_yaxes(tickformat=".0%", title=None, rangemode="tozero")
            fig.update_xaxes(dtick=1, title=None)
            show(style(fig, 380))
        with right:
            st.markdown("**By company and year**")
            grid = theme.pivot(index="company", columns="year", values="share")
            grid = grid.loc[grid[LAST].sort_values(ascending=False).index]
            grid.columns = [str(c) for c in grid.columns]
            show(heatmap(grid, 560))

        st.markdown(f"**Example passages** · {LAST}")
        examples = passages[(passages["year"] == LAST) & passages["text"].str.contains(pattern)]
        examples = examples[~examples["category"].isin(NON_RISK)]
        for _, row in examples.drop_duplicates("company").head(4).iterrows():
            with st.expander(f"{row['company']} · {row['category']}"):
                st.markdown(pattern.sub(lambda m: f"**{m.group(0)}**", row["text"]))

    st.divider()
    st.markdown("**Search by meaning**")
    st.caption("Describe a risk in your own words. The search compares meaning (embeddings), not exact words.")
    question = st.text_input("Describe a risk", placeholder="for example: running out of cash, or a key supplier going bankrupt")
    if question.strip():
        try:
            vector = get_embedder().encode([question.strip()])[0]
            scores = load_embeddings() @ vector
            hits = passages.assign(score=scores).sort_values("score", ascending=False)
            hits = hits.drop_duplicates(["company", "category"]).head(6)
            for _, row in hits.iterrows():
                with st.expander(f"{row['company']} {row['year']} · {row['category']} · match {row['score']:.2f}"):
                    st.write(row["text"])
        except Exception as exc:  # the model download can fail on a cold start
            st.warning(f"The search model could not be loaded right now ({type(exc).__name__}). Please try again.")

# ================================================================ LIVE
with tab_live:
    st.markdown(
        "Type **any US stock ticker**. The app downloads the company's newest annual report from the SEC, "
        "finds the risk section, and classifies it while you watch. Nothing here is precomputed."
    )
    with st.form("live"):
        c1, c2 = st.columns([3, 1])
        ticker = c1.text_input("Ticker", value="NFLX", max_chars=8, help="Examples: NFLX, AMD, DIS, COST, UBER, PEP, INTC")
        c2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        run = c2.form_submit_button("Analyse", type="primary", width="stretch")

    if run and ticker.strip():
        from riskradar.analyze import analyze_filing
        from riskradar.extract import ExtractionError
        from riskradar.sec import SecError

        symbol = ticker.strip().upper()
        try:
            started = time.perf_counter()
            with st.status(f"Running the pipeline for {symbol}…", expanded=True) as status:
                embedder = get_embedder()
                result = analyze_filing(
                    symbol, get_sec_client(), embedder, get_classifier(),
                    on_step=lambda name, seconds, detail: st.write(f"✅ **{name}** · {seconds:.1f}s · {detail}"),
                )
                status.update(label=f"Done in {time.perf_counter() - started:.1f} seconds", state="complete")
            st.session_state["live_result"] = result
        except (SecError, ExtractionError) as exc:
            st.session_state.pop("live_result", None)
            st.error(str(exc))
        except Exception as exc:
            st.session_state.pop("live_result", None)
            st.error(f"Something went wrong ({type(exc).__name__}). SEC or the model host may be busy. Please try again.")

    result = st.session_state.get("live_result")
    if result is not None:
        filing, mix = result.filing, result.mix
        st.subheader(f"{filing.sec_name} · fiscal year {filing.fiscal_year}")
        st.caption(f"[Open the filing on SEC EDGAR]({filing.url}) · filed {filing.filed}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Words in risk section", f"{result.words:,}")
        m2.metric("Passages", len(result.passages))
        top = mix.idxmax()
        m3.metric("Biggest risk type", top, f"{mix[top]:.0%} of the section", delta_color="off", delta_arrow="off")

        left, right = st.columns([6, 5])
        with left:
            st.markdown(f"**Risk mix vs the {info['companies']}-company average**")
            order = mix.sort_values().index
            fig = go.Figure()
            fig.add_bar(x=benchmark[order], y=order, orientation="h", name=f"Average of {info['companies']} companies ({LAST})",
                        marker_color=GREY, marker_cornerradius=3, hovertemplate="%{y}: %{x:.1%}<extra>average</extra>")
            fig.add_bar(x=mix[order], y=order, orientation="h", name=filing.ticker, marker_color=BLUE, marker_cornerradius=3,
                        text=[f"{v:.0%}" for v in mix[order]], textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: <b>%{x:.1%}</b><extra>" + filing.ticker + "</extra>")
            fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.08), bargap=0.25)
            fig.update_xaxes(tickformat=".0%", title=None)
            show(style(fig, 520))
        with right:
            diff = (mix - benchmark).sort_values()
            st.markdown("**What stands out**")
            st.write(
                f"Compared with the average company, **{filing.ticker}** spends more of its risk section on "
                f"**{diff.index[-1]}** ({diff.iloc[-1] * 100:+.0f} points) and **{diff.index[-2]}** "
                f"({diff.iloc[-2] * 100:+.0f} points), and less on **{diff.index[0]}** ({diff.iloc[0] * 100:+.0f} points)."
            )
            mentions = {name: int(result.passages["text"].str.contains(p).sum()) for name, p in THEMES.items()}
            st.markdown("**Theme mentions** (passages)")
            st.dataframe(
                pd.DataFrame({"Theme": list(mentions), "Passages": list(mentions.values())}).sort_values("Passages", ascending=False),
                hide_index=True, width="stretch", height=260,
            )
        st.markdown("**Every passage and its label**")
        st.dataframe(
            result.passages[["category", "confidence", "second", "text"]],
            hide_index=True, width="stretch", height=360,
            column_config={
                "category": st.column_config.TextColumn("Risk type", width="medium"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                "second": st.column_config.TextColumn("Second guess", width="medium"),
                "text": st.column_config.TextColumn("Passage", width="large"),
            },
        )

# ================================================================ METHOD
with tab_method:
    st.subheader("How it works")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f6f8fb", color="#b9bcc2", fontname="Helvetica", fontsize=11];
          edge [color="#52514e"];
          sec [label="SEC EDGAR\\n10-K filing (HTML)"]; extract [label="Find Item 1A\\nRisk Factors"];
          chunk [label="Split into\\n~170-word passages"]; embed [label="Embed\\n(MiniLM, ONNX)"];
          classify [label="Classify\\n(logistic regression)"]; stats [label="Shares, trends,\\nthemes, reuse"]; app [label="This app"];
          sec -> extract -> chunk -> embed -> classify -> stats -> app;
        }
        """
    )
    st.markdown(
        f"""
1. **Download** each 10-K from SEC EDGAR (free official API, no key).
2. **Extract** the real *Item 1A. Risk Factors* section. The same words also appear in the table of contents and in cross-references, so only headings at the start of a line count, and the longest candidate wins.
3. **Split** it into passages of about 170 words along sentence borders, sized to fit the model's reading window.
4. **Embed** each passage with `{info['embedding_model'].split('/')[-1]}`, a small model that runs without a GPU. The same model is used offline and in the live tab.
5. **Classify** with logistic regression trained on labelled passages. Only the weights are stored, so predicting needs NumPy alone.
6. **Measure** shares instead of counts, so a longer report does not look riskier just because it is longer.
"""
    )

    st.subheader("How good is the classifier?")
    low, high = metrics["test_accuracy_ci95"]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Accuracy", f"{metrics['test_accuracy']:.1%}", f"95% range {low:.0%}–{high:.0%}", delta_color="off", delta_arrow="off")
    a2.metric("Right answer in top 2", f"{metrics['test_top2_accuracy']:.0%}")
    a3.metric("Macro F1", f"{metrics['test_macro_f1']:.2f}")
    a4.metric("Baseline (most common class)", f"{metrics['baseline_most_common_class']:.0%}")
    st.markdown(
        f"""
- **{metrics['train_passages'] + metrics['test_passages'] + metrics['train_removed_as_near_copies_of_test']} passages were labelled** following a written
  [labelling guide](https://github.com/kishan244/risk-narrative-nlp/blob/main/data/eval/LABELING_GUIDE.md).
  **{metrics['test_passages']} of them are a random test set** (10 per company) that was never used to train or tune anything.
- 10-K filings repeat paragraphs year after year. **{metrics['train_removed_as_near_copies_of_test']} training passages that were near-copies of a test passage were removed**, so the model is never tested on text it has already seen.
- The first labelling round (339 training passages) scored 67%. Labelling 328 more, picked where the model was least sure, raised it to {metrics['test_accuracy']:.1%}. No setting was chosen using the test set.
"""
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**F1 score per risk type**")
        per = pd.DataFrame(metrics["per_category"]).sort_values("f1")
        fig = px.bar(per, x="f1", y="category", orientation="h", text_auto=".2f", color_discrete_sequence=[BLUE],
                     hover_data={"precision": ":.2f", "recall": ":.2f", "test_passages": True})
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(range=[0, 1], title=None)
        fig.update_yaxes(title=None)
        show(style(fig, 440))
        st.caption("Hover for precision, recall and the number of test passages. Types with few test passages have noisy scores.")
    with right:
        st.markdown("**Where it gets confused**")
        st.caption("Rows: true label · Columns: model's label · Numbers: test passages")
        labels = metrics["confusion"]["labels"]
        short = [name.split(" & ")[0].split(" / ")[0] for name in labels]
        fig = px.imshow(np.array(metrics["confusion"]["matrix"]), x=short, y=short, text_auto=True,
                        color_continuous_scale=SEQUENTIAL, aspect="auto")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(xgap=2, ygap=2, hovertemplate="true: %{y}<br>model: %{x}<br><b>%{z}</b><extra></extra>")
        fig.update_xaxes(tickangle=45, title=None)
        fig.update_yaxes(title=None)
        show(style(fig, 440))

    st.markdown("**Accuracy per sector**")
    sectors = pd.Series(metrics["per_sector_accuracy"]).rename("Accuracy").to_frame().T
    st.dataframe(sectors.style.format("{:.0%}"), width="stretch")

    st.subheader("Limits")
    st.markdown(
        """
- **One label per passage.** About one passage in four covers two risks. That is why the top-2 score is much higher than the accuracy.
- **Energy companies are hard.** Their risk sections pack several topics into each passage.
- **20 large US companies** is a sample, not the market.
- **More text is not more risk.** The app measures what companies *write about*, not how likely a risk is.
- Not investment advice.
"""
    )
    st.markdown(
        f"Code and data: [github.com/kishan244/risk-narrative-nlp](https://github.com/kishan244/risk-narrative-nlp) · "
        f"version {info['version']} · data built {info['built']}"
    )
