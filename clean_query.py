from fastapi import FastAPI, Query, HTTPException
from typing import Literal, Optional, List
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import logging, uvicorn
from aws import download_from_s3
import math
from fastapi.middleware.cors import CORSMiddleware



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDINGS_DIR = "static/new_big_embed"
MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 20 # number of results per dataset
MIN_RESULTS_PER_DATASET = 1  # Minimum results to try to get from each dataset
MAX_RESULTS_PER_DATASET = 8  # Maximum results from a single dataset

print("Loading embedding model...")
try:
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise


def prepare_all_faiss_dbs(local_dir="static/new_big_embed"):
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
        # if not os.path.exists(local_index):
        #     print(f"Downloading {index_file} from S3...")
        #     download_from_s3(f"{s3_path}/{index_file}", local_index)
        # else:
        #     print(f"{index_file} already exists locally, skipping download.")

        # if not os.path.exists(local_pkl):
        #     print(f"Downloading {pkl_file} from S3...")
        #     download_from_s3(f"{s3_path}/{pkl_file}", local_pkl)
        # else:
        #     print(f"{pkl_file} already exists locally, skipping download.")

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

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def mmr(doc_embeddings, query_embedding, top_n=10, lambda_param=0.7):
    selected = []
    doc_indices = list(range(len(doc_embeddings)))
    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array(doc_embeddings)

    relevance_scores = cosine_similarity(doc_embeddings, query_embedding).reshape(-1)

    while len(selected) < top_n and doc_indices:
        if not selected:
            # Pick the most relevant document first
            idx = np.argmax(relevance_scores)
            selected.append(idx)
            doc_indices.remove(idx)
        else:
            # MMR formula
            max_score = -float("inf")
            selected_embs = doc_embeddings[selected]
            for i in doc_indices:
                sim_to_query = relevance_scores[i]
                sim_to_selected = np.max(cosine_similarity(doc_embeddings[i].reshape(1, -1), selected_embs))
                mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
                if mmr_score > max_score:
                    max_score = mmr_score
                    selected_idx = i
            selected.append(selected_idx)
            doc_indices.remove(selected_idx)

    return selected


def search_dataset(dataset_name, data, prompt_embedding, mode, k, extract_fields=None):
    """Search a single dataset with diversity using MMR"""
    try:
        index = data["index"]
        mapping = data["mapping"]
        
        if mode not in mapping:
            logger.warning(f"Mode '{mode}' not found in dataset '{dataset_name}'")
            return []
            
        indices = mapping[mode]["indices"]
        metadata_list = mapping[mode]["metadata"]

        if len(indices) == 0:
            logger.info(f"No {mode} entries found in dataset '{dataset_name}'")
            return []

        search_k = min(k * 2, len(indices))  # search more for diversity
        D, I = index.search(np.array([prompt_embedding]), search_k)

        # Filter and collect valid entries
        valid_items = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1 or idx not in indices:
                continue
            try:
                actual_idx = list(indices).index(idx)
                if actual_idx < len(metadata_list):
                    metadata = clean_metadata(metadata_list[actual_idx])
                    clean_score = clean_float_value(score)
                    if clean_score is None:
                        continue
                    valid_items.append((idx, clean_score, metadata))
            except Exception as e:
                logger.warning(f"Error processing idx {idx}: {e}")

        if len(valid_items) == 0:
            return []

        # Generate embeddings for these metadata texts (assumes model globally available)
        texts = [item[2].get("text", "") for item in valid_items]
        embeddings = model.encode(texts)

        # Apply MMR
        selected_indices = mmr(embeddings, prompt_embedding, top_n=k)

        # Final results using selected indices
        results = []
        for sel_idx in selected_indices:
            idx, score, metadata = valid_items[sel_idx]
            if extract_fields:
                extracted_data = {field: metadata.get(field) for field in extract_fields if field in metadata}
            else:
                extracted_data = metadata

            results.append({
                "dataset": dataset_name,
                "score": score,
                "data": extracted_data
            })

        return results

    except Exception as e:
        logger.error(f"Error searching dataset '{dataset_name}': {e}")
        return []

def balanced_search(prompt_embedding, mode, total_results=TOP_K, extract_fields=None):
    """Perform balanced search across all datasets"""
    # Get available datasets that have data for the given mode
    available_datasets = []
    for dataset_name, data in index_data.items():
        mapping = data["mapping"]
        if mode in mapping and len(mapping[mode]["indices"]) > 0:
            available_datasets.append(dataset_name)
    
    if not available_datasets:
        return []
    
    # Calculate initial allocation per dataset
    datasets_count = len(available_datasets)
    base_per_dataset = max(MIN_RESULTS_PER_DATASET, total_results // datasets_count)
    
    # First round: Get minimum results from each dataset
    all_results = []
    dataset_results = {}
    
    for dataset_name in available_datasets:
        data = index_data[dataset_name]
        results = search_dataset(dataset_name, data, prompt_embedding, mode, base_per_dataset, extract_fields)
        dataset_results[dataset_name] = results
        all_results.extend(results)
    
    # If we don't have enough results, do a second round to fill up
    current_total = len(all_results)
    if current_total < total_results:
        remaining_needed = total_results - current_total
        
        # Sort datasets by how many results they returned (to prioritize datasets with more relevant content)
        dataset_result_counts = [(len(dataset_results[name]), name) for name in available_datasets]
        dataset_result_counts.sort(reverse=True)  # Datasets with more results first
        
        # Distribute remaining slots
        for _, dataset_name in dataset_result_counts:
            if remaining_needed <= 0:
                break
                
            current_count = len(dataset_results[dataset_name])
            
            # Don't exceed MAX_RESULTS_PER_DATASET from any single dataset
            can_add = min(
                remaining_needed,
                MAX_RESULTS_PER_DATASET - current_count,
                10  # Don't add too many at once
            )
            
            if can_add > 0:
                data = index_data[dataset_name]
                additional_results = search_dataset(
                    dataset_name, data, prompt_embedding, mode, 
                    current_count + can_add, extract_fields
                )[current_count:]  # Get only the new results
                
                all_results.extend(additional_results)
                remaining_needed -= len(additional_results)
    
    # Sort all results by score and limit to requested number
    all_results = sorted(all_results, key=lambda x: x["score"])[:total_results]
    
    # Log the distribution for debugging
    result_distribution = {}
    for result in all_results:
        dataset = result["dataset"]
        result_distribution[dataset] = result_distribution.get(dataset, 0) + 1
    
    logger.info(f"Result distribution: {result_distribution}")
    
    return all_results

def format_grouped_results(results, extract_fields=None, primary_field=None):
    """Format results grouped by dataset with extracted values"""
    grouped_results = {}
    
    for result in results:
        dataset = result["dataset"]
        data = result["data"]
        
        if dataset not in grouped_results:
            grouped_results[dataset] = []
        
        # If a primary field is specified, extract just that value
        if primary_field and primary_field in data:
            grouped_results[dataset].append(data[primary_field])
        # If extract_fields is specified and has only one field, extract just that value
        elif extract_fields and len(extract_fields) == 1 and extract_fields[0] in data:
            grouped_results[dataset].append(data[extract_fields[0]])
        # Otherwise, append the entire data object
        else:
            grouped_results[dataset].append(data)
    
    return grouped_results

@app.get("/")
def root():
    return {
        "message": "Embedding Search API",
        "available_datasets": list(index_data.keys()),
        "total_datasets": len(index_data)
    }

@app.get("/fields")
def get_available_fields(
    dataset: Optional[str] = Query(None, description="Specific dataset name to get fields for"),
    mode: Optional[Literal["student", "professional"]] = Query(None, description="Mode to get fields for")
):
    """Get available fields/columns from datasets"""
    try:
        field_info = {}
        
        datasets_to_check = [dataset] if dataset else list(index_data.keys())
        
        for dataset_name in datasets_to_check:
            if dataset_name not in index_data:
                continue
                
            data = index_data[dataset_name]
            mapping = data["mapping"]
            
            field_info[dataset_name] = {}
            
            modes_to_check = [mode] if mode else ["student", "professional"]
            
            for mode_name in modes_to_check:
                if mode_name not in mapping:
                    continue
                    
                metadata_list = mapping[mode_name]["metadata"]
                
                if not metadata_list:
                    field_info[dataset_name][mode_name] = []
                    continue
                
                # Get fields from first few entries to understand structure
                sample_size = min(5, len(metadata_list))
                all_fields = set()
                
                for i in range(sample_size):
                    if i < len(metadata_list) and isinstance(metadata_list[i], dict):
                        all_fields.update(metadata_list[i].keys())
                
                field_info[dataset_name][mode_name] = sorted(list(all_fields))
        
        return {
            "available_fields": field_info,
            "usage_examples": {
                "extract_single_field": "?fields=title&primary_field=title",
                "extract_multiple_fields": "?fields=title,author,url",
                "grouped_format": "?format_type=grouped&fields=title",
                "detailed_format": "?format_type=detailed&fields=title,description"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting field information: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving field information")

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
    mode: Literal["student", "professional"] = Query(..., description="Entry type"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to extract (e.g., 'title,author,url')"),
    primary_field: Optional[str] = Query(None, description="Primary field to extract as the main value for each result"),
    format_type: Literal["grouped", "detailed"] = Query("grouped", description="Response format: 'grouped' for dataset-grouped results, 'detailed' for full metadata")
):
    try:
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Search prompt cannot be empty")
        
        # Parse extract fields
        extract_fields = None
        if fields:
            extract_fields = [field.strip() for field in fields.split(",") if field.strip()]
        
        # Generate embedding for the prompt
        try:
            prompt_embedding = model.encode([prompt.strip()])[0].astype(np.float32)
            
            # Check for invalid embeddings
            if np.any(np.isnan(prompt_embedding)) or np.any(np.isinf(prompt_embedding)):
                raise HTTPException(status_code=500, detail="Generated embedding contains invalid values")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(status_code=500, detail="Error processing search prompt")

        # Use balanced search
        results = balanced_search(prompt_embedding, mode, TOP_K, extract_fields)
        
        # Count datasets searched
        datasets_searched = len(set(result["dataset"] for result in results))

        # Format response based on requested format
        if format_type == "grouped":
            formatted_results = format_grouped_results(results, extract_fields, primary_field)
            
            return {
                "query": prompt.strip(),
                "mode": mode,
                "datasets_searched": datasets_searched,
                "total_results": len(results),
                "extracted_fields": extract_fields,
                "primary_field": primary_field,
                "results": formatted_results
            }
        else:
            # Detailed format (original format)
            detailed_results = []
            for result in results:
                detailed_results.append({
                    "dataset": result["dataset"],
                    "score": result["score"],
                    "data": result["data"]
                })
            
            return {
                "query": prompt.strip(),
                "mode": mode,
                "datasets_searched": datasets_searched,
                "total_results": len(results),
                "extracted_fields": extract_fields,
                "results": detailed_results
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