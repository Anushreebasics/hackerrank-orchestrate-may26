import pandas as pd
import os
import argparse
from dotenv import load_dotenv
from retriever import Retriever
from agent import SupportAgent
from ingest import load_and_embed_corpus

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate Support Agent")
    parser.add_argument("--input", default="../support_tickets/support_tickets.csv", help="Input CSV path")
    parser.add_argument("--output", default="../support_tickets/output.csv", help="Output CSV path")
    args = parser.parse_args()

    kb_path = "knowledge_base.pkl"
    
    if not os.path.exists(kb_path):
        print("Knowledge base not found. Generating embeddings (this may take a minute)...")
        load_and_embed_corpus("../data")
        
    print("Loading Retriever and Agent...")
    retriever = Retriever(kb_path)
    agent = SupportAgent()
    
    print(f"Reading input from {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input}")
        return
        
    results = []
    
    for idx, row in df.iterrows():
        # Handle cases where column names might be slightly different
        issue = str(row.get('Issue', row.get('issue', '')))
        subject = str(row.get('Subject', row.get('subject', '')))
        company = str(row.get('Company', row.get('company', 'None')))
        
        print(f"Processing ticket {idx+1}/{len(df)}")
        
        # 1. Retrieve
        query = issue + " " + subject
        retrieved_chunks = retriever.search(query, top_k=5, company_filter=company)
        
        # 2. Reason & Generate
        triage_output = agent.generate_response(issue, subject, company, retrieved_chunks)
        
        results.append({
            "status": triage_output.status,
            "product_area": triage_output.product_area,
            "response": triage_output.response,
            "justification": triage_output.justification,
            "request_type": triage_output.request_type
        })
        
    # Create final dataframe
    out_df = pd.DataFrame(results)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Finished processing. Output written to {args.output}")

if __name__ == "__main__":
    main()
