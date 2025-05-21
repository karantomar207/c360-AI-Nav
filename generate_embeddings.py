import pandas as pd
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

# Ensure the embeddings directory exists
os.makedirs('embeddings', exist_ok=True)

# Load the CSV file
print("Loading data from CSV...")
df = pd.read_csv('data/dummy_data.csv')
print(f"Loaded {len(df)} entries from CSV.")

# Initialize the sentence transformer model
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the skills
print("Generating embeddings for skills...")
texts = df['skill'].tolist()
embeddings = model.encode(texts, show_progress_bar=True)
print(f"Generated {len(embeddings)} embeddings.")

# Create and save the FAISS index
print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"Added {index.ntotal} vectors to the index.")

# Save the FAISS index
print("Saving FAISS index...")
faiss.write_index(index, 'embeddings/faiss_index.bin')

# Save the metadata
print("Saving metadata...")
with open('embeddings/meta.pkl', 'wb') as f:
    pickle.dump(df.to_dict(orient='records'), f)

print("Embedding generation completed successfully!")
print(f"Index saved to: embeddings/faiss_index.bin")
print(f"Metadata saved to: embeddings/meta.pkl")