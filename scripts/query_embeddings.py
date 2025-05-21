import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('embeddings/faiss_index.bin')

with open('embeddings/meta.pkl', 'rb') as f:
    meta_data = pickle.load(f)

def search(prompt, top_k=3):
    embedding = model.encode([prompt])
    D, I = index.search(embedding, top_k)
    
    # Get the top results
    results = [meta_data[i] for i in I[0]]
    
    # Format the response to match the frontend's expected structure
    if results:
        # Initialize response with empty lists for each category
        response = {
            "jobs": [],
            "certifications": [],
            "youtube_channels": [],
            "ebooks": [],
            "websites": []
        }
        
        # Populate the response from the results
        for result in results:
            # Extract jobs (comma-separated string to list)
            if "jobs" in result:
                jobs = [job.strip() for job in result["jobs"].split(",")]
                response["jobs"].extend(jobs)
            
            # Extract certifications (comma-separated string to list)
            if "certifications" in result:
                certifications = [cert.strip() for cert in result["certifications"].split(",")]
                response["certifications"].extend(certifications)
            
            # Extract YouTube channels (comma-separated string to list)
            if "youtube_channels" in result:
                channels = [channel.strip() for channel in result["youtube_channels"].split(",")]
                response["youtube_channels"].extend(channels)
            
            # Extract ebooks (comma-separated string to list)
            if "ebooks" in result:
                books = [book.strip() for book in result["ebooks"].split(",")]
                response["ebooks"].extend(books)
            
            # Extract websites (comma-separated string to list)
            if "websites" in result:
                sites = [site.strip() for site in result["websites"].split(",")]
                response["websites"].extend(sites)
        
        # Remove duplicates and return only the top 5 items per category
        for key in response:
            response[key] = list(dict.fromkeys(response[key]))[:5]
        
        return response
    
    return {
        "jobs": [],
        "certifications": [],
        "youtube_channels": [],
        "ebooks": [],
        "websites": []
    }

if __name__ == "__main__":
    prompt = input("Enter a skill or interest: ")
    results = search(prompt)
    print("Top job recommendations:", results["jobs"])
    print("Recommended certifications:", results["certifications"])
    print("YouTube channels to follow:", results["youtube_channels"])
    print("Recommended ebooks:", results["ebooks"])
    print("Useful websites:", results["websites"])