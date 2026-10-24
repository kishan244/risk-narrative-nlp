import os
import re
import pdfplumber
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from config import *

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_pdf_text(pdf_path):
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()

            if text:
                full_text += text + "\n"

    return full_text


# -----------------------------
# Extract Risk Factors section
# -----------------------------
def extract_risk_section(text):
    # Normalize spacing because annual reports use different line breaks and heading styles
    normalized_text = re.sub(r"\s+", " ", text)

    # Common Item 1A heading patterns across Apple, JPMorgan, and Tesla 10-K PDFs
    start_patterns = [
        r"Item\s+1A\.?\s+Risk\s+Factors",
        r"ITEM\s+1A\.?\s+RISK\s+FACTORS",
        r"Item\s+1A\s*[-–—:]\s*Risk\s+Factors",
        r"Risk\s+Factors"
    ]

    start_matches = []
    for pattern in start_patterns:
        for match in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
            start_matches.append(match.start())

    if not start_matches:
        return None

    # Many reports mention Item 1A in the table of contents first.
    # Use the second match when available; otherwise use the first real match.
    start_matches = sorted(set(start_matches))
    if len(start_matches) >= 2:
        start = start_matches[1]
    else:
        start = start_matches[0]

    # Common next section patterns after Item 1A
    end_patterns = [
        r"Item\s+1B\.?\s+Unresolved\s+Staff\s+Comments",
        r"ITEM\s+1B\.?\s+UNRESOLVED\s+STAFF\s+COMMENTS",
        r"Item\s+1C\.?\s+Cybersecurity",
        r"ITEM\s+1C\.?\s+CYBERSECURITY",
        r"Item\s+2\.?\s+Properties",
        r"ITEM\s+2\.?\s+PROPERTIES"
    ]

    end = len(normalized_text)
    search_area = normalized_text[start + 100:]

    for pattern in end_patterns:
        match = re.search(pattern, search_area, flags=re.IGNORECASE)
        if match:
            candidate_end = start + 100 + match.start()
            end = min(end, candidate_end)

    risk_text = normalized_text[start:end].strip()

    # Reject very short false matches from table of contents
    if len(risk_text.split()) < 500:
        if len(start_matches) >= 3:
            start = start_matches[2]
            end = len(normalized_text)
            search_area = normalized_text[start + 100:]
            for pattern in end_patterns:
                match = re.search(pattern, search_area, flags=re.IGNORECASE)
                if match:
                    candidate_end = start + 100 + match.start()
                    end = min(end, candidate_end)
            risk_text = normalized_text[start:end].strip()

    if len(risk_text.split()) < 500:
        return None

    return risk_text


# -----------------------------
# Split into chunks
# -----------------------------
def split_into_chunks(text):
    words = text.split()
    chunks = []

    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i:i + CHUNK_SIZE])

        if len(chunk.split()) >= MIN_WORDS_PER_CHUNK:
            chunks.append(chunk)

    return chunks


# -----------------------------
# Process PDFs
# -----------------------------
def process_pdfs():
    documents = []
    years = []
    companies = []

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("\nSearching PDFs in:", RAW_PDF_FOLDER)

    for company in COMPANIES:
        def normalize_name(name):
            return re.sub(r"[^a-z0-9]", "", name.lower())

        folder_aliases = {
            "apple": ["apple", "aapl"],
            "jpmorgan": ["jpmorgan", "jpmorganchase", "jp", "jpm", "chase"],
            "tesla": ["tesla", "tsla"]
        }

        company_folder = None
        available_folders = [
            folder for folder in os.listdir(RAW_PDF_FOLDER)
            if os.path.isdir(os.path.join(RAW_PDF_FOLDER, folder))
        ]

        for folder in available_folders:
            normalized_folder = normalize_name(folder)
            aliases = folder_aliases.get(company, [company])

            if any(alias in normalized_folder for alias in aliases):
                company_folder = os.path.join(RAW_PDF_FOLDER, folder)
                break

        if company_folder is None:
            print(f"Folder not found for company: {company}")
            print(f"Available folders inside {RAW_PDF_FOLDER}: {available_folders}")
            print(f"Expected folder name to contain one of: {folder_aliases.get(company, [company])}")
            continue

        print(f"\nCompany folder: {company_folder}")

        for year in YEARS:
            expected_file = f"{company}_{year}.pdf"
            pdf_path = os.path.join(company_folder, expected_file)
            file = expected_file

            if not os.path.exists(pdf_path):
                matching_files = [
                    f for f in os.listdir(company_folder)
                    if f.lower().endswith(".pdf") and year in f
                ]

                if matching_files:
                    file = matching_files[0]
                    pdf_path = os.path.join(company_folder, file)
                else:
                    print(f"Missing PDF for {company.upper()} {year}. Expected: {pdf_path}")
                    continue

            print(f"\nProcessing: {company.upper()} {year} -> {file}")

            text = extract_pdf_text(pdf_path)
            risk_text = extract_risk_section(text)

            if risk_text:
                chunks = split_into_chunks(risk_text)
                original_chunk_count = len(chunks)

                # Balance the corpus so long JPMorgan reports do not dominate the topic model.
                # Keep the first MAX_CHUNKS_PER_REPORT chunks from each report because Risk Factors
                # sections usually list the most material risks near the beginning.
                if len(chunks) > MAX_CHUNKS_PER_REPORT:
                    chunks = chunks[:MAX_CHUNKS_PER_REPORT]

                for chunk in chunks:
                    documents.append(chunk)
                    years.append(year)
                    companies.append(company)

                out_file = f"{company}_{year}_risk.txt"
                with open(os.path.join(OUTPUT_FOLDER, out_file), "w", encoding="utf-8") as f:
                    f.write(risk_text)

                print(f"Saved risk text: {out_file}")
                print(f"Original chunks: {original_chunk_count}")
                print(f"Balanced chunks used: {len(chunks)}")

            else:
                print(f"Risk section not found in {file}")

    return documents, years, companies


def run_topic_modeling(documents, years, companies):
    print("\nLoading AI model...")

    embedding_model = SentenceTransformer(MODEL_NAME)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        min_samples=3,
        metric="euclidean",
        prediction_data=True
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics=NR_TOPICS,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(documents)

    # Save topic info
    topic_info = topic_model.get_topic_info()

    # ---------------------------------
    # Automatic topic labeling
    # ---------------------------------
    category_keywords = {
        "Cybersecurity & Privacy Risk": [
            "data", "privacy", "security", "attacks", "confidential", "information", "authentication"
        ],
        "Supply Chain Risk": [
            "components", "supply", "manufacturing", "partners", "outsourcing", "assembly", "shortages"
        ],
        "Operational Risk": [
            "operations", "facilities", "business", "disruptions", "systems"
        ],
        "Competition Risk": [
            "competitors", "markets", "competition", "products", "services"
        ],
        "Financial & Tax Risk": [
            "tax", "stock", "financial", "price", "investment", "revenue"
        ],
        "Currency & International Risk": [
            "foreign", "currency", "dollar", "currencies", "exchange"
        ],
        "Product Quality Risk": [
            "defects", "quality", "products", "issues", "failures"
        ],
        "Trade & Vendor Risk": [
            "trade", "vendor", "credit", "receivables", "terms"
        ]
    }

    auto_labels = []
    used_labels = {}

    for _, row in topic_info.iterrows():
        topic_words = row["Representation"]

        if isinstance(topic_words, list):
            words = [w.lower() for w in topic_words]
        else:
            words = str(topic_words).lower().replace("[", "").replace("]", "").replace("'", "").split(", ")

        scores = {}

        for category, keywords in category_keywords.items():
            score = len(set(words) & set(keywords))
            scores[category] = score

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        # if nothing meaningful matched
        if best_score == 0:
            best_label = "Unclassified Risk"

        # avoid duplicate label names
        if best_label in used_labels:
            used_labels[best_label] += 1
            best_label = f"{best_label} ({used_labels[best_label]})"
        else:
            used_labels[best_label] = 1

        auto_labels.append(best_label)

    topic_info["Auto_Label"] = auto_labels

    topic_info.to_csv(f"{OUTPUT_FOLDER}/final_topic_info.csv", index=False)

    # Save chunk-level topics
    df = pd.DataFrame({
        "company": companies,
        "year": years,
        "text_chunk": documents,
        "topic": topics
    })

    df.to_csv(f"{OUTPUT_FOLDER}/final_chunk_topics.csv", index=False)

    # Temporal analysis
    df = df[df["topic"] != -1]

    temporal = df.groupby(["company", "year", "topic"]).size().reset_index(name="count")
    temporal.to_csv(f"{OUTPUT_FOLDER}/final_temporal_topics.csv", index=False)

    print("\nSaved:")
    print("final_topic_info.csv")
    print("final_chunk_topics.csv")
    print("final_temporal_topics.csv")

