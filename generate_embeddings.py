import pandas as pd
import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup directories
DATA_DIR = "data"
EMBEDDINGS_DIR = "embeddings"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load high-quality embedding model
print("Loading BGE Base embedding model...")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')  # High-accuracy 768-dim model

# Classification keywords
PROFESSIONAL_KWS = ['senior', 'lead', 'manager', 'expert', 'experienced', 'advanced', 
                    'principal', 'director', 'head', 'chief', 'architect', '5+ years',
                    'masters required', 'phd required', 'management', 'leadership',
                    'strategic', 'enterprise', 'executive']

STUDENT_KWS = ['entry', 'junior', 'trainee', 'intern', 'fresher', 'beginner',
               'basic', 'introduction', 'fundamentals', 'starter', 'new grad',
               'graduate', 'bachelor', 'student', 'learn', 'course']

def classify_student(row):
    text = ' '.join(map(str, row.values)).lower()
    pro = sum(1 for k in PROFESSIONAL_KWS if k in text)
    stu = sum(1 for k in STUDENT_KWS if k in text)
    return stu >= pro

def create_faiss_index(data, prefix):
    print(f"\nCreating embeddings for: {prefix}")
    texts = data.apply(lambda row: ' '.join(map(str, row.values)), axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings).astype('float32')

    # Sanity check for dimensions
    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss_path = f"{EMBEDDINGS_DIR}/{prefix}_faiss_index.bin"
    meta_path = f"{EMBEDDINGS_DIR}/{prefix}_meta.pkl"

    faiss.write_index(index, faiss_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(data.to_dict(orient='records'), f)

    print(f"Saved {prefix} index and metadata.")

def generate_suggestions(df):
    print("\nGenerating suggestions from careers_data.csv...")
    suggestions = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_text = ' '.join(map(str, row.values)).lower()

        base_skill = next((kw for kw in STUDENT_KWS + PROFESSIONAL_KWS if kw in row_text), "general")
        additional_skills = ', '.join([kw for kw in row_text.split() if kw not in base_skill])[:100]
        career_path = row.get("role", "Unknown Role")
        timeline = "6-12 months" if classify_student(row) else "1-2 years"
        description = row.get("description", "")[:200]
        field = row.get("field", "Technology")
        level = "student" if classify_student(row) else "professional"

        suggestions.append({
            "base_skill": base_skill,
            "additional_skills": additional_skills,
            "career_path": career_path,
            "timeline": timeline,
            "description": description,
            "field": field,
            "experience_level": level
        })

    suggestions_df = pd.DataFrame(suggestions)
    suggestions_df.to_csv(f"{DATA_DIR}/career_suggestions.csv", index=False)
    print(f"Saved career suggestions to career_suggestions.csv")
    return suggestions_df

def main():
    careers_path = f"{DATA_DIR}/careers_data.csv"
    if not os.path.exists(careers_path):
        print(f"Error: {careers_path} not found!")
        return

    df = pd.read_csv(careers_path)
    print(f"Loaded {len(df)} rows from careers_data.csv")

    # Classify data
    df['for_students'] = df.apply(classify_student, axis=1)
    student_df = df[df['for_students']].drop(columns=['for_students'])
    professional_df = df[~df['for_students']].drop(columns=['for_students'])

    student_df.to_csv(f"{DATA_DIR}/student_data.csv", index=False)
    professional_df.to_csv(f"{DATA_DIR}/professional_data.csv", index=False)
    print("Saved student and professional data.")

    # Create embeddings
    if not student_df.empty:
        create_faiss_index(student_df, "student")
    if not professional_df.empty:
        create_faiss_index(professional_df, "professional")

    # Generate and embed suggestions
    suggestions_df = generate_suggestions(df)
    if not suggestions_df.empty:
        create_faiss_index(suggestions_df, "suggestions")

    print("\nAll embeddings and suggestions generated successfully.")

if __name__ == "__main__":
    main()
