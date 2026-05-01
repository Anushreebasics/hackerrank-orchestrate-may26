import os
import glob
import pickle
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def load_and_embed_corpus(data_dir):
    md_files = glob.glob(os.path.join(data_dir, '**', '*.md'), recursive=True)
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    documents = []
    for filepath in md_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            parts = filepath.split(os.sep)
            try:
                data_idx = parts.index("data")
                company = parts[data_idx + 1] if len(parts) > data_idx + 1 else "Unknown"
            except ValueError:
                company = "Unknown"
                
            md_header_splits = markdown_splitter.split_text(content)
            chunks = text_splitter.split_documents(md_header_splits)
            
            for chunk in chunks:
                if len(chunk.page_content.strip()) < 10:
                    continue
                    
                meta_str = ", ".join([f"{k}: {v}" for k, v in chunk.metadata.items()])
                if meta_str:
                    enriched_text = f"[{meta_str}]\n{chunk.page_content.strip()}"
                else:
                    enriched_text = chunk.page_content.strip()
                    
                documents.append({
                    "text": enriched_text,
                    "company": company,
                    "filepath": filepath
                })
                
    print(f"Loaded {len(documents)} chunks. Generating Dense Embeddings locally...")
    
    texts = [doc["text"] for doc in documents]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_matrix = model.encode(texts, convert_to_tensor=False)
    
    # Save the documents and the embeddings
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump({
            "documents": documents,
            "embeddings_matrix": embeddings_matrix
        }, f)
        
    print("Saved knowledge base to knowledge_base.pkl")

if __name__ == "__main__":
    load_and_embed_corpus("../data")
