import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Configuration
EMBEDDINGS_DIR = "static/gap_embeddings"
CSV_FILE = "static/gap_csv_data/jobs_data.csv"  # Make sure to update if path changes
INDEX_FILE = "jobs.index"
METADATA_FILE = "jobs.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_data(csv_path):
    return pd.read_csv(csv_path)

def preprocess_text(row):
    # Combine title, description, and skills into one string
    text = f"{row['title']} - {row['description']} Skills: {row['skills']}"
    return text

def create_embeddings(texts, model):
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype('float32')

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index_and_metadata(index, metadata_dict, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    faiss.write_index(index, os.path.join(dir_path, INDEX_FILE))
    with open(os.path.join(dir_path, METADATA_FILE), "wb") as f:
        pickle.dump(metadata_dict, f)

def main():
    print("Loading data...")
    df = load_data(CSV_FILE)
    print(f"Loaded {len(df)} job records.")

    print("Preprocessing text...")
    texts = df.apply(preprocess_text, axis=1).tolist()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    embeddings = create_embeddings(texts, model)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving index and metadata...")
    metadata_dict = df.to_dict(orient="index")  # {0: {...}, 1: {...}, ...}
    save_index_and_metadata(index, metadata_dict, EMBEDDINGS_DIR)

    print("Embedding process completed successfully.")

if __name__ == "__main__":
    main()
