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
