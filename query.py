from fastapi import FastAPI, Query, HTTPException
from typing import Literal
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import logging, uvicorn
from aws import download_from_s3


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

EMBEDDINGS_DIR = "static/aws_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 20 # number of results per dataset

print("Loading embedding model...")
try:
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise


def prepare_all_faiss_dbs(local_dir="static/aws_embeddings"):
    os.makedirs(local_dir, exist_ok=True)

    s3_path = os.getenv("FAISS_DB_PATH")
    if not s3_path:
        raise ValueError("Environment variable FAISS_DB_PATH is not set")

    # List all your DB base names here (without extensions)
    db_names = ["ebooks", "jobs", 'courses', 'certificates']  # Add more DB names as needed

    all_indexes = {}
    all_metadata = {}

    for db_name in db_names:
        print(f"\nPreparing FAISS DB: {db_name}")

        index_file = f"{db_name}.index"
        pkl_file = f"{db_name}.pkl"

        local_index = os.path.join(local_dir, index_file)
        local_pkl = os.path.join(local_dir, pkl_file)

        # Download files only if not present locally
        if not os.path.exists(local_index):
            print(f"Downloading {index_file} from S3...")
            download_from_s3(f"{s3_path}/{index_file}", local_index)
        else:
            print(f"{index_file} already exists locally, skipping download.")

        if not os.path.exists(local_pkl):
            print(f"Downloading {pkl_file} from S3...")
            download_from_s3(f"{s3_path}/{pkl_file}", local_pkl)
        else:
            print(f"{pkl_file} already exists locally, skipping download.")

        # Load the FAISS index and metadata
        index = faiss.read_index(local_index)
        with open(local_pkl, 'rb') as f:
            metadata = pickle.load(f)

        all_indexes[db_name] = index
        all_metadata[db_name] = metadata

    return all_indexes, all_metadata

print("Loading indexes and metadata from S3...")
try:
    index_data = {}
    indexes, metadatas = prepare_all_faiss_dbs()
    for key in indexes:
        # Assuming your pickle metadata has "mode_mapping" as before
        index_data[key] = {
            "index": indexes[key],
            "mapping": metadatas[key]["mode_mapping"]
        }
    print(f"✓ Successfully loaded {len(index_data)} datasets: {list(index_data.keys())}")
except Exception as e:
    print(f"❌ Error loading datasets: {e}")
    raise


index_data = {}

try:
    if not os.path.exists(EMBEDDINGS_DIR):
        raise FileNotFoundError(f"Embeddings directory '{EMBEDDINGS_DIR}' not found")
    
    files_found = os.listdir(EMBEDDINGS_DIR)
    index_files = [f for f in files_found if f.endswith(".index")]
    
    if not index_files:
        raise FileNotFoundError(f"No .index files found in '{EMBEDDINGS_DIR}'")
    
    for file in index_files:
        key = file.replace(".index", "")
        index_path = os.path.join(EMBEDDINGS_DIR, file)
        pkl_path = os.path.join(EMBEDDINGS_DIR, f"{key}.pkl")

        if not os.path.exists(pkl_path):
            logger.warning(f"Metadata file {pkl_path} not found, skipping {key}")
            continue

        try:
            index = faiss.read_index(index_path)
            with open(pkl_path, "rb") as f:
                metadata = pickle.load(f)

            index_data[key] = {
                "index": index,
                "mapping": metadata["mode_mapping"]
            }
            print(f"✓ Loaded dataset: {key}")
            
        except Exception as e:
            logger.error(f"Error loading {key}: {e}")
            continue

    if not index_data:
        raise RuntimeError("No datasets could be loaded")
        
    print(f"✓ Successfully loaded {len(index_data)} datasets: {list(index_data.keys())}")
    
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    raise

def clean_float_value(value):
    """Clean float values to ensure JSON compatibility"""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        if abs(value) > 1e308:  # Very large values that might cause JSON issues
            return None
        return float(value)
    return value

def clean_metadata(metadata):
    """Clean metadata dictionary to ensure all values are JSON serializable"""
    cleaned = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            cleaned[k] = clean_metadata(v)
        elif isinstance(v, list):
            cleaned[k] = [clean_float_value(item) for item in v]
        else:
            cleaned[k] = clean_float_value(v)
    return cleaned

@app.get("/")
def root():
    return {
        "message": "Embedding Search API",
        "available_datasets": list(index_data.keys()),
        "total_datasets": len(index_data)
    }

@app.get("/datasets")
def get_datasets():
    """Get information about available datasets"""
    dataset_info = {}
    for name, data in index_data.items():
        mapping = data["mapping"]
        dataset_info[name] = {
            "student_entries": len(mapping["student"]["indices"]),
            "professional_entries": len(mapping["professional"]["indices"]),
            "total_entries": len(mapping["student"]["indices"]) + len(mapping["professional"]["indices"])
        }
    return dataset_info

@app.get("/search")
def search(
    prompt: str = Query(..., description="Search prompt", min_length=1),
    mode: Literal["student", "professional"] = Query(..., description="Entry type")
):
    try:
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Search prompt cannot be empty")
        
        # Generate embedding for the prompt
        try:
            prompt_embedding = model.encode([prompt.strip()])[0].astype(np.float32)
            
            # Check for invalid embeddings
            if np.any(np.isnan(prompt_embedding)) or np.any(np.isinf(prompt_embedding)):
                raise HTTPException(status_code=500, detail="Generated embedding contains invalid values")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(status_code=500, detail="Error processing search prompt")

        results = []
        datasets_searched = 0

        for dataset_name, data in index_data.items():
            try:
                index = data["index"]
                mapping = data["mapping"]
                
                if mode not in mapping:
                    logger.warning(f"Mode '{mode}' not found in dataset '{dataset_name}'")
                    continue
                    
                indices = mapping[mode]["indices"]
                metadata_list = mapping[mode]["metadata"]

                if len(indices) == 0:
                    logger.info(f"No {mode} entries found in dataset '{dataset_name}'")
                    continue

                datasets_searched += 1
                
                # Perform the search
                search_k = min(TOP_K * 3, len(indices))  # Search for more results initially
                D, I = index.search(np.array([prompt_embedding]), search_k)
                
                # Process results
                for score, idx in zip(D[0], I[0]):
                    if idx == -1:  # Invalid index returned by FAISS
                        continue
                        
                    if idx in indices:
                        try:
                            actual_idx = list(indices).index(idx)
                            if actual_idx < len(metadata_list):
                                metadata = metadata_list[actual_idx]
                                cleaned_metadata = clean_metadata(metadata)
                                
                                # Clean the score value
                                clean_score = clean_float_value(score)
                                if clean_score is None:
                                    continue
                                
                                results.append({
                                    "dataset": dataset_name,
                                    "score": clean_score,
                                    "metadata": cleaned_metadata
                                })
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error processing result from {dataset_name}: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error searching dataset '{dataset_name}': {e}")
                continue

        # Sort results by score (lower is better for L2 distance) and limit
        results = sorted(results, key=lambda x: x["score"])[:TOP_K]

        return {
            "query": prompt.strip(),
            "mode": mode,
            "datasets_searched": datasets_searched,
            "total_results": len(results),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "datasets_loaded": len(index_data),
        "embedding_dimension": model.get_sentence_embedding_dimension() if model else None
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)