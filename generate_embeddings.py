import pandas as pd
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Set up directories
DATA_DIR = "data"
EMBEDDINGS_DIR = "embeddings"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Initialize SentenceTransformer model
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate and save FAISS index
def create_and_save_embeddings(data, prefix):
    print(f"\nGenerating embeddings for {prefix} data...")

    # Use all columns to form a single string per row
    texts = data.apply(lambda row: ' '.join(map(str, row.values)), axis=1).tolist()

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Create and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss_path = f"{EMBEDDINGS_DIR}/{prefix}_faiss_index.bin"
    meta_path = f"{EMBEDDINGS_DIR}/{prefix}_meta.pkl"

    faiss.write_index(index, faiss_path)

    with open(meta_path, 'wb') as f:
        pickle.dump(data.to_dict(orient='records'), f)

    print(f"FAISS index saved to: {faiss_path}")
    print(f"Metadata saved to: {meta_path}")

def main():
    # Load full dataset
    print("Loading career data...")
    all_data = pd.read_csv(f"{DATA_DIR}/careers_data.csv")

    # Classify for students vs professionals
    print("Classifying entries...")
    def classify(row):
        keywords = ['senior', 'lead', 'manager', 'expert', 'experienced', 'advanced', 'principal']
        text = ' '.join(map(str, row.values)).lower()
        if any(kw in text for kw in keywords):
            return False
        return True

    all_data['for_students'] = all_data.apply(classify, axis=1)

    # Split datasets
    student_data = all_data[all_data['for_students']].copy()
    professional_data = all_data[~all_data['for_students']].copy()

    student_data.to_csv(f"{DATA_DIR}/student_data.csv", index=False)
    professional_data.to_csv(f"{DATA_DIR}/professional_data.csv", index=False)

    # Create and save embeddings
    create_and_save_embeddings(student_data, "student")
    create_and_save_embeddings(professional_data, "professional")

    print("\nEmbedding generation completed successfully!")

if __name__ == "__main__":
    main()
