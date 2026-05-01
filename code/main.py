import pandas as pd
import os
import argparse
from dotenv import load_dotenv
from retriever import Retriever
from agent import SupportAgent, is_valid_company
from ingest import load_and_embed_corpus
from semantic_cache import SemanticCache

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
    # Semantic cache to avoid repeated LLM calls for near-duplicate queries
    semantic_cache = SemanticCache(cache_path="semantic_cache.pkl", model=retriever.bi_encoder)
    
    print(f"Reading input from {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input}")
        return
        
    results = []

    def is_cacheable_response(triage_output):
        """Only cache successful, non-error LLM responses."""
        if triage_output.request_type == "invalid":
            return False
        if triage_output.response.strip().lower().startswith("error processing request"):
            return False
        if "llm service error" in triage_output.justification.strip().lower():
            return False
        return True
    
    for idx, row in df.iterrows():
        # Handle cases where column names might be slightly different
        issue = str(row.get('Issue', row.get('issue', '')))
        subject = str(row.get('Subject', row.get('subject', '')))
        company = str(row.get('Company', row.get('company', 'None')))
        
        print(f"Processing ticket {idx+1}/{len(df)}")
        
        # Build a query string and check semantic cache first
        # (skip cache lookup if company is invalid; let agent escalate immediately)
        query = issue + " " + subject

        if is_valid_company(company):
            cached = semantic_cache.get_cached_response(query, threshold=0.98)
            if cached is not None:
                print(f"Cache hit for ticket {idx+1}; using cached response.")
                # Ensure replied responses begin with "Hi,"
                resp_text = cached.get("response", "")
                if cached.get("status", "").lower() == "replied":
                    rt = resp_text.strip()
                    if not rt.lower().startswith(("hi", "hello")):
                        resp_text = "Hi, " + rt

                response_dict = {
                    "issue": issue,
                    "subject": subject,
                    "company": company,
                    "response": resp_text,
                    "product_area": cached.get("product_area", ""),
                    "status": cached.get("status", ""),
                    "request_type": cached.get("request_type", ""),
                    "justification": cached.get("justification", "")
                }
                results.append(response_dict)
                continue

        # 1. Retrieve (only if company is valid)
        if is_valid_company(company):
            retrieved_chunks = retriever.search(query, top_k=5, company_filter=company)
        else:
            retrieved_chunks = []

        # 2. Reason & Generate
        triage_output = agent.generate_response(issue, subject, company, retrieved_chunks)

        response_dict = {
            "issue": issue,
            "subject": subject,
            "company": company,
            "response": triage_output.response,
            "product_area": triage_output.product_area,
            "status": triage_output.status,
            "request_type": triage_output.request_type,
            "justification": triage_output.justification
        }

        results.append(response_dict)

        # Store this query + response in the semantic cache for future reuse
        # (only if company is valid and status is not an error)
        if is_valid_company(company) and is_cacheable_response(triage_output):
            try:
                # Cache stores only the agent-generated fields, not input columns
                cache_dict = {
                    "response": ("Hi, " + triage_output.response.strip()) if triage_output.status.lower() == "replied" and not triage_output.response.strip().lower().startswith(("hi", "hello")) else triage_output.response,
                    "product_area": triage_output.product_area,
                    "status": triage_output.status,
                    "request_type": triage_output.request_type,
                    "justification": triage_output.justification,
                    "chain_of_thought": triage_output.chain_of_thought
                }
                semantic_cache.add(query, cache_dict)
            except Exception:
                # Cache failures shouldn't break processing
                pass
        
    # Create final dataframe
    out_df = pd.DataFrame(results)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Finished processing. Output written to {args.output}")

if __name__ == "__main__":
    main()
