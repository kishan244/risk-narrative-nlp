import pandas as pd

def generate_report():
    topic_info = pd.read_csv("outputs/final_topic_info.csv")
    temporal = pd.read_csv("outputs/final_temporal_topics.csv")

    topic_info = topic_info[topic_info["Topic"] != -1]
    temporal = temporal[temporal["topic"] != -1]

    topic_info = topic_info[
    ~topic_info["Auto_Label"].str.contains("Unclassified Risk", na=False)
]

    topic_info["Clean_Label"] = topic_info["Auto_Label"].str.replace(r" \(\d+\)", "", regex=True)

    merged = temporal.merge(
        topic_info[["Topic", "Clean_Label"]],
        left_on="topic",
        right_on="Topic",
        how="inner"
    )

    grouped = merged.groupby(["company", "year", "Clean_Label"])["count"].sum().reset_index()

    overall_grouped = grouped.groupby(["year", "Clean_Label"])["count"].sum().reset_index()
    pivot = overall_grouped.pivot(index="year", columns="Clean_Label", values="count").fillna(0)

    report = []

    report.append("=== FINAL AI RISK ANALYSIS REPORT ===\n")

    report.append("Method Summary:\n")
    report.append(
        "The system extracted Risk Factors sections from annual report PDFs, "
        "split the text into semantic chunks, generated transformer-based sentence embeddings, "
        "applied BERTopic for topic discovery, and analyzed topic frequency across reporting years.\n"
    )

    report.append("Key Findings:\n")

    trend_data = []

    for label in pivot.columns:
        values = pivot[label].tolist()
        years = pivot.index.tolist()

        start = values[0]
        end = values[-1]
        change = end - start

        peak_year = years[values.index(max(values))]
        peak_value = max(values)

        if change > 0:
            trend = "increasing"
        elif change < 0:
            trend = "decreasing"
        else:
            trend = "stable"

        trend_data.append({
            "label": label,
            "start": start,
            "end": end,
            "change": change,
            "trend": trend,
            "peak_year": peak_year,
            "peak_value": peak_value
        })

    increasing = sorted([x for x in trend_data if x["trend"] == "increasing"], key=lambda x: x["change"], reverse=True)
    decreasing = sorted([x for x in trend_data if x["trend"] == "decreasing"], key=lambda x: x["change"])
    stable = [x for x in trend_data if x["trend"] == "stable"]

    if increasing:
        top_inc = increasing[0]
        report.append(
            f"- Strongest increasing theme: {top_inc['label']} "
            f"({int(top_inc['start'])} → {int(top_inc['end'])})."
        )

    if decreasing:
        top_dec = decreasing[0]
        report.append(
            f"- Strongest decreasing theme: {top_dec['label']} "
            f"({int(top_dec['start'])} → {int(top_dec['end'])})."
        )

    all_peaks = sorted(trend_data, key=lambda x: x["peak_value"], reverse=True)
    strongest_peak = all_peaks[0]

    report.append(
        f"- Highest observed topic intensity: {strongest_peak['label']} "
        f"in {strongest_peak['peak_year']} with frequency {int(strongest_peak['peak_value'])}."
    )

    report.append("\nDetailed Topic Trends:\n")

    for item in trend_data:
        label = item["label"]
        start = item["start"]
        end = item["end"]
        trend = item["trend"]
        peak_year = item["peak_year"]

        if trend == "increasing":
            interpretation = (
                f"{label} became more prominent over time, indicating stronger emphasis "
                f"on this risk area in later annual reports."
            )
        elif trend == "decreasing":
            interpretation = (
                f"{label} declined over time, indicating reduced relative emphasis "
                f"within the risk narrative."
            )
        else:
            interpretation = (
                f"{label} remained stable, suggesting consistent attention "
                f"throughout the reporting period."
            )

        report.append(f"{label}")
        report.append(f"  Start year value: {int(start)}")
        report.append(f"  End year value: {int(end)}")
        report.append(f"  Trend: {trend}")
        report.append(f"  Peak year: {peak_year}")
        report.append(f"  Interpretation: {interpretation}\n")

    report.append("Overall Summary:\n")

    if increasing:
        report.append("- Increasing themes: " + ", ".join([x["label"] for x in increasing]) + ".")
    if decreasing:
        report.append("- Decreasing themes: " + ", ".join([x["label"] for x in decreasing]) + ".")
    if stable:
        report.append("- Stable themes: " + ", ".join([x["label"] for x in stable]) + ".")

    report.append("\nAcademic Conclusion:\n")
    report.append(
        "The analysis demonstrates that transformer-based topic modeling can identify "
        "semantic shifts in corporate risk narratives over time. The results show that "
        "risk disclosures are not static; instead, different risk themes increase, decrease, "
        "or stabilize across reporting years. This supports the use of NLP-driven temporal "
        "topic modeling as a method for analyzing annual report risk disclosures."
    )

    report_text = "\n".join(report)

    with open("outputs/final_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    grouped.to_csv("outputs/final_labeled_topic_trends.csv", index=False)

    print("\nSaved: outputs/final_report.txt")
    print("Saved: outputs/final_labeled_topic_trends.csv")