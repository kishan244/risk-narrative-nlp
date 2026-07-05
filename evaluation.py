import pandas as pd

def generate_evaluation_report():
    topic_info = pd.read_csv("outputs/final_topic_info.csv")
    chunks = pd.read_csv("outputs/final_chunk_topics.csv")
    trends = pd.read_csv("outputs/final_labeled_topic_trends.csv")

    total_chunks = len(chunks)
    outliers = len(chunks[chunks["topic"] == -1])
    usable_chunks = total_chunks - outliers

    outlier_rate = round((outliers / total_chunks) * 100, 2)

    topic_info = topic_info[topic_info["Topic"] != -1]
    num_topics = len(topic_info)

    labeled_topics = topic_info[
    ~topic_info["Auto_Label"].str.contains("Unclassified Risk", na=False)
    ]

    label_coverage = round((len(labeled_topics) / num_topics) * 100, 2)

    years = sorted(trends["year"].unique())
    num_years = len(years)

    unique_labels = trends["Clean_Label"].nunique()

    report = []

    report.append("=== MODEL EVALUATION REPORT ===\n")

    report.append("Evaluation Metrics:\n")
    report.append(f"Total text chunks analyzed: {total_chunks}")
    report.append(f"Usable clustered chunks: {usable_chunks}")
    report.append(f"Outlier chunks: {outliers}")
    report.append(f"Outlier rate: {outlier_rate}%")
    report.append(f"Number of discovered topics: {num_topics}")
    report.append(f"Automatically labeled topic coverage: {label_coverage}%")
    report.append(f"Number of reporting years analyzed: {num_years}")
    report.append(f"Number of final risk themes: {unique_labels}\n")

    report.append("Interpretation:\n")

    if outlier_rate < 15:
        report.append("- The outlier rate is low, indicating that most text chunks were successfully assigned to meaningful topics.")
    elif outlier_rate < 30:
        report.append("- The outlier rate is moderate, indicating acceptable but improvable topic assignment.")
    else:
        report.append("- The outlier rate is high, indicating that topic modeling parameters may need further tuning.")

    if label_coverage >= 80:
        report.append("- Topic labeling coverage is strong, showing that most discovered topics were mapped to interpretable risk categories.")
    else:
        report.append("- Topic labeling coverage is limited, suggesting that more topic labels or domain-specific keywords may be needed.")

    report.append(
        "- The model produces interpretable temporal risk themes, supporting its use for semantic evolution analysis."
    )

    report.append("\nLimitations:\n")
    report.append("- The model is unsupervised, so topic quality depends on clustering and representation quality.")
    report.append("- Topic labels are generated using an automatic keyword-based mapping layer.")
    report.append("- Results should be interpreted as analytical support, not as legal or financial advice.")

    report_text = "\n".join(report)

    with open("outputs/model_evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Saved: outputs/model_evaluation_report.txt")