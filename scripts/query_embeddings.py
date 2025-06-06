import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Path to data files and embeddings
student_index = faiss.read_index('static/embeddings/student_faiss_index.bin')
with open('static/embeddings/student_meta.pkl', 'rb') as f:
    student_meta = pickle.load(f)

professional_index = faiss.read_index('static/embeddings/professional_faiss_index.bin')
with open('static/embeddings/professional_meta.pkl', 'rb') as f:
    professional_meta = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')


def search(prompt, mode='student', top_k=5):

    # Get query embedding
    embedding = model.encode([prompt], convert_to_numpy=True)

    # Select appropriate index and meta_data
    if mode == 'professional':
        index = professional_index
        meta_data = professional_meta
    else:
        index = student_index
        meta_data = student_meta

    # Perform search
    D, I = index.search(embedding, top_k)
    results = [meta_data[i] for i in I[0] if i < len(meta_data)]

    print("resultsresultsresults", results)

    # Initialize structured response
    response = {
        "jobs": [],
        "certifications": [],
        "youtube_channels": [],
        "ebooks": [],
        "websites": []
    }

    # Extract and split fields properly
    for item in results:
        if 'jobs' in item:
            jobs = [j.strip() for j in item['jobs'].split(',')]
            response['jobs'].extend(jobs)

        if 'certifications' in item:
            certs = [c.strip() for c in item['certifications'].split(',')]
            response['certifications'].extend(certs)

        if 'youtube_channels' in item:
            channels = [y.strip() for y in item['youtube_channels'].split(',')]
            response['youtube_channels'].extend(channels)

        if 'ebooks' in item:
            books = [b.strip() for b in item['ebooks'].split(',')]
            response['ebooks'].extend(books)

        if 'websites' in item:
            sites = [s.strip() for s in item['websites'].split(',')]
            response['websites'].extend(sites)

    # Deduplicate and return only top 5
    for key in response:
        response[key] = list(dict.fromkeys(response[key]))[:5]

    print("response", response)
    return response



if __name__ == "__main__":
    prompt = input("Enter a skill or interest: ")
    user_type = input("Enter mode (student/professional): ").strip().lower()
    results = search(prompt, mode=user_type)
    
    print("\nTop job recommendations:", results["jobs"])
    print("Recommended certifications:", results["certifications"])
    print("YouTube channels to follow:", results["youtube_channels"])
    print("Recommended ebooks:", results["ebooks"])
    print("Useful websites:", results["websites"])
