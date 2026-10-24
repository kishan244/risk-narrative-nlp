from pipeline import process_pdfs, run_topic_modeling
from report_generator import generate_report
from final_visuals import create_visuals
from semantic_shift import generate_semantic_shift_report
from evaluation import generate_evaluation_report

def main():
    print("\n=== Corporate Risk Narrative Analysis System ===")

    # Step 1: Process PDFs
    documents, years, companies = process_pdfs()

    if not documents:
        print("No valid documents found.")
        return

    print(f"\nTotal chunks prepared: {len(documents)}")

    # Step 2: Run AI topic modeling
    run_topic_modeling(documents, years, companies)

    # Step 3: Generate final report
    generate_report()
    create_visuals()
    generate_semantic_shift_report()
    generate_evaluation_report()
     
    print("\n=== Analysis Complete ===")
    print("Check outputs folder for results.")

if __name__ == "__main__":
    main()