import os
import glob
import pickle
from sentence_transformers import SentenceTransformer

def chunk_text(text, max_chars=1500):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) < max_chars:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def load_and_embed_corpus(data_dir):
    md_files = glob.glob(os.path.join(data_dir, '**', '*.md'), recursive=True)
    
    documents = []
    for filepath in md_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = chunk_text(content)
            
            parts = filepath.split(os.sep)
            try:
                data_idx = parts.index("data")
                company = parts[data_idx + 1] if len(parts) > data_idx + 1 else "Unknown"
            except ValueError:
                company = "Unknown"
                
            for chunk in chunks:
                if len(chunk.strip()) < 10:
                    continue
                documents.append({
                    "text": chunk,
                    "company": company,
                    "filepath": filepath
                })
                
    print(f"Loaded {len(documents)} chunks. Generating Dense Embeddings locally...")
    
    texts = [doc["text"] for doc in documents]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_matrix = model.encode(texts, convert_to_tensor=False)
    
    # Save the documents and the fitted model
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump({
            "documents": documents,
            "embeddings_matrix": embeddings_matrix
        }, f)
        
    print("Saved knowledge base to knowledge_base.pkl")

if __name__ == "__main__":
    load_and_embed_corpus("../data")
