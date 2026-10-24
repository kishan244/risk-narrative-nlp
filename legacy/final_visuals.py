import pandas as pd
import matplotlib.pyplot as plt

def create_visuals():
    df = pd.read_csv("outputs/final_labeled_topic_trends.csv")
    df = df[df["Clean_Label"] != "Unclassified Risk"]

    overall = df.groupby(["year", "Clean_Label"])["count"].sum().reset_index()
    pivot = overall.pivot(index="year", columns="Clean_Label", values="count").fillna(0)

    # Overall multi-topic evolution chart
    plt.figure(figsize=(12, 7))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker='o', label=col)

    plt.title("Temporal Evolution of Corporate Risk Themes")
    plt.xlabel("Year")
    plt.ylabel("Topic Frequency")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/final_topic_evolution_chart.png", dpi=300)
    plt.close()

    # Overall heatmap
    plt.figure(figsize=(12, 7))
    plt.imshow(pivot.T, aspect='auto')
    plt.yticks(range(len(pivot.columns)), pivot.columns)
    plt.xticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Frequency")
    plt.title("Risk Theme Intensity Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/final_topic_heatmap.png", dpi=300)
    plt.close()

    # Company comparison chart
    company_totals = df.groupby(["company", "Clean_Label"])["count"].sum().reset_index()
    company_pivot = company_totals.pivot(index="Clean_Label", columns="company", values="count").fillna(0)

    plt.figure(figsize=(12, 7))
    company_pivot.plot(kind="bar", figsize=(12, 7))
    plt.title("Risk Theme Comparison by Company")
    plt.xlabel("Risk Theme")
    plt.ylabel("Topic Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/company_risk_theme_comparison.png", dpi=300)
    plt.close()

    # Company-year total chunk trend
    company_year = df.groupby(["company", "year"])["count"].sum().reset_index()
    company_year_pivot = company_year.pivot(index="year", columns="company", values="count").fillna(0)

    plt.figure(figsize=(10, 6))
    for col in company_year_pivot.columns:
        plt.plot(company_year_pivot.index, company_year_pivot[col], marker='o', label=col)

    plt.title("Risk Narrative Volume by Company Over Time")
    plt.xlabel("Year")
    plt.ylabel("Risk Topic Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/company_risk_volume_trend.png", dpi=300)
    plt.close()

    print("Saved:")
    print("outputs/final_topic_evolution_chart.png")
    print("outputs/final_topic_heatmap.png")
    print("outputs/company_risk_theme_comparison.png")
    print("outputs/company_risk_volume_trend.png")