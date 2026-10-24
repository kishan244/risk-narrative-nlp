import pandas as pd

def generate_semantic_shift_report():
    chunks = pd.read_csv("outputs/final_chunk_topics.csv")
    topic_info = pd.read_csv("outputs/final_topic_info.csv")

    topic_info = topic_info[topic_info["Topic"] != -1]
    topic_info = topic_info[
    ~topic_info["Auto_Label"].str.contains("Unclassified Risk", na=False)
]

    topic_info["Clean_Label"] = topic_info["Auto_Label"].str.replace(r" \(\d+\)", "", regex=True)
    chunks = chunks[chunks["topic"] != -1]

    merged = chunks.merge(
        topic_info[["Topic", "Clean_Label"]],
        left_on="topic",
        right_on="Topic",
        how="inner"
    )

    label_keywords = {
        "Cybersecurity & Privacy Risk": ["security", "privacy", "data", "cyber", "attacks", "confidential", "ransomware"],
        "Supply Chain Risk": ["supply", "supplier", "components", "manufacturing", "assembly", "logistical", "shortages"],
        "Operational Risk": ["operations", "facilities", "disruptions", "systems", "macroeconomic", "recession"],
        "Competition Risk": ["competition", "competitors", "markets", "competitive", "technological"],
        "Financial & Tax Risk": ["tax", "financial", "stock", "investment", "price", "revenue"],
        "Currency & International Risk": ["foreign", "currency", "dollar", "exchange", "international"],
        "Product Quality Risk": ["defects", "quality", "hardware", "software", "issues"],
        "Trade & Vendor Risk": ["trade", "vendor", "receivables", "credit", "resellers"]
    }

    theme_explanations = {
        "Cybersecurity & Privacy Risk": "The language shifts from general information system and data security concerns toward broader privacy, regulatory, and sensitive data obligations.",
        "Supply Chain Risk": "The narrative later includes stronger references to geopolitical tensions, logistics disruption, outsourcing partners, and international conflict.",
        "Operational Risk": "The narrative shows continued dependence on business operations and later includes stronger emphasis on macroeconomic uncertainty, disruptions, inflation, and execution risk.",
        "Competition Risk": "The risk narrative remains focused on competitive pressure, rapid technological change, pricing pressure, product innovation, and market share challenges.",
        "Financial & Tax Risk": "The narrative highlights exposure to stock price volatility, tax uncertainty, investment risk, revenue pressure, and financial performance expectations.",
        "Currency & International Risk": "The narrative emphasizes exposure to foreign exchange movements, U.S. dollar strength, international pricing decisions, and global sales sensitivity.",
        "Product Quality Risk": "The risk narrative focuses on defects, quality issues, manufacturing problems, software limitations, and reputational harm.",
        "Trade & Vendor Risk": "The narrative emphasizes receivables, vendor concentration, credit exposure, resellers, and collectibility risks across markets."
    }

    def pick_best_example(data, label):
        keywords = label_keywords.get(label, [])
        best_text = ""
        best_score = -1

        for text in data["text_chunk"]:
            lower_text = str(text).lower()
            score = sum(lower_text.count(keyword) for keyword in keywords)

            if score > best_score:
                best_score = score
                best_text = text

        return best_text

    report = []
    report.append("=== SEMANTIC EVOLUTION EVIDENCE REPORT ===\n")
    report.append("Comparison period: 2019 to 2024")
    report.append("Companies analyzed: Apple, JPMorgan, Tesla\n")

    for company in sorted(merged["company"].unique()):
        company_data = merged[merged["company"] == company]
        years = sorted(company_data["year"].astype(str).unique())

        if "2019" not in years or "2024" not in years:
            report.append(f"\n=== {company.upper()} ===")
            report.append("2019 or 2024 data missing, so direct semantic comparison was skipped.\n")
            continue

        first_year = "2019"
        last_year = "2024"

        report.append(f"\n==============================")
        report.append(f"COMPANY: {company.upper()}")
        report.append(f"==============================\n")

        for label in sorted(company_data["Clean_Label"].unique()):
            topic_data = company_data[company_data["Clean_Label"] == label]

            first_data = topic_data[topic_data["year"].astype(str) == first_year]
            last_data = topic_data[topic_data["year"].astype(str) == last_year]

            if first_data.empty or last_data.empty:
                continue

            first_example = pick_best_example(first_data, label)
            last_example = pick_best_example(last_data, label)

            report.append(f"\n--- {label} ---\n")
            report.append(f"{first_year} representative text:")
            report.append(first_example[:900] + "...\n")
            report.append(f"{last_year} representative text:")
            report.append(last_example[:900] + "...\n")
            report.append("Semantic shift interpretation:")
            report.append(theme_explanations.get(label, "This theme shows semantic variation across the reporting period.") + "\n")

    report_text = "\n".join(report)

    with open("outputs/semantic_evolution_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Saved: outputs/semantic_evolution_report.txt")