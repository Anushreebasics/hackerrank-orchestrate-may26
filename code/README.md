# Support Triage Agent

This is the implementation of the terminal-based support triage agent for the HackerRank Orchestrate hackathon.

## Architecture

1. **Ingestion (`ingest.py`)**: Reads the Markdown corpus from `../data`, chunks it by paragraphs, and vectorizes it locally into Dense Semantic Embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`) to bypass embedding API limits. The vectors are stored locally in `knowledge_base.pkl`.
2. **Two-Stage Retrieval (`retriever.py`)**: Implements an advanced retrieval pipeline:
   - *Stage 1 (Bi-Encoder)*: A fast local cosine-similarity search using the dense embedding matrix to fetch the top 20 candidates.
   - *Stage 2 (Cross-Encoder)*: A highly accurate re-ranking of those candidates using a Cross-Encoder neural network (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to ensure maximum relevance.
3. **Reasoning (`agent.py`)**: Uses Google's `gemini-2.5-flash` via the `google-genai` SDK with Pydantic Structured Outputs to guarantee that the output matches the required schema (`status`, `product_area`, `response`, `justification`, `request_type`).
4. **Orchestration (`main.py`)**: Glues everything together, reads the input CSV, runs the pipeline per row, and writes the output CSV.

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file and add your Gemini API Key:
   ```bash
   cp .env.example .env
   # Edit .env and add GEMINI_API_KEY=your_key
   ```

## Usage

To test against the sample data:
```bash
python main.py --input ../support_tickets/sample_support_tickets.csv --output ../support_tickets/sample_output.csv
```

To run the final evaluation:
```bash
python main.py --input ../support_tickets/support_tickets.csv --output ../support_tickets/output.csv
```

## Upgrades

This project includes three recent upgrades to improve accuracy, reliability, and efficiency:

### 1. Semantic Cache

Before calling the LLM, the pipeline now embeds the incoming query using the retriever's bi-encoder and checks for a cached response with cosine similarity >= 0.98. A cache hit returns the exact previously generated response (no LLM call). The cache is persisted to `semantic_cache.pkl` in the `code/` folder. 

**To clear the cache and force fresh LLM calls:**

```bash
rm code/semantic_cache.pkl
```

### 2. Self-Reflection / Chain-of-Thought

The Pydantic `TriageOutput` schema now includes a `chain_of_thought` field. The LLM is instructed to populate this field with a numbered, step-by-step internal scratchpad BEFORE setting the final `status`. The CSV output includes `chain_of_thought` as the last column.

### 3. Improved Triage Accuracy

- **Company Validation**: Tickets without a valid company field are immediately escalated as "unknown product" instead of being passed to the LLM.
- **Retry Logic**: The agent now retries LLM calls with exponential backoff when it encounters `503 UNAVAILABLE` errors from the API.
- **Response Normalization**: Cached responses are normalized before writing to CSV to ensure `chain_of_thought` is always the final column.

## Usage Notes

- The cache uses the same local bi-encoder model as the retriever (`all-MiniLM-L6-v2`) to compute embeddings.
- You can adjust the cache similarity threshold in `code/main.py` where `get_cached_response(query, threshold=0.98)` is called.
- Run unit tests (no API calls required):

```bash
python test_agent_logic.py
```

- Example run (writes `chain_of_thought` as the final column):

```bash
python main.py --input ../support_tickets/sample_support_tickets.csv --output ../support_tickets/sample_output_final.csv
```
