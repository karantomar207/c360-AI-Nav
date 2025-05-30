from fastapi import FastAPI, Query, HTTPException
from typing import Literal, Optional, List, Dict, Any
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import logging, uvicorn
from aws import download_from_s3
import math
from groq import Groq
import json
from pydantic import BaseModel
import asyncio
from datetime import datetime
import re
from fastapi.middleware.cors import CORSMiddleware




# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Vector Search API with Content Enhancement", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMBEDDINGS_DIR = "static/new_big_embed"
MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 20
MIN_RESULTS_PER_DATASET = 1
MAX_RESULTS_PER_DATASET = 8

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. Content enhancement features will be disabled.")
    groq_client = None
else:
    groq_client = Groq(api_key=GROQ_API_KEY)

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

    db_names = ["ebooks", "jobs", 'courses', 'certificates']
    all_indexes = {}
    all_metadata = {}

    for db_name in db_names:
        print(f"\nPreparing FAISS DB: {db_name}")
        index_file = f"{db_name}.index"
        pkl_file = f"{db_name}.pkl"
        local_index = os.path.join(local_dir, index_file)
        local_pkl = os.path.join(local_dir, pkl_file)

        index = faiss.read_index(local_index)
        with open(local_pkl, 'rb') as f:
            metadata = pickle.load(f)

        all_indexes[db_name] = index
        all_metadata[db_name] = metadata

    return all_indexes, all_metadata

# Load indexes
print("Loading indexes and metadata...")
try:
    index_data = {}
    
    if not os.path.exists(EMBEDDINGS_DIR):
        indexes, metadatas = prepare_all_faiss_dbs()
        for key in indexes:
            index_data[key] = {
                "index": indexes[key],
                "mapping": metadatas[key]["mode_mapping"]
            }
    else:
        files_found = os.listdir(EMBEDDINGS_DIR)
        index_files = [f for f in files_found if f.endswith(".index")]
        
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
        if abs(value) > 1e308:
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

def mmr(doc_embeddings, query_embedding, top_n=10, lambda_param=0.7):
    selected = []
    doc_indices = list(range(len(doc_embeddings)))
    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array(doc_embeddings)

    relevance_scores = cosine_similarity(doc_embeddings, query_embedding).reshape(-1)

    while len(selected) < top_n and doc_indices:
        if not selected:
            idx = np.argmax(relevance_scores)
            selected.append(idx)
            doc_indices.remove(idx)
        else:
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

        search_k = min(k * 2, len(indices))
        D, I = index.search(np.array([prompt_embedding]), search_k)

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

        texts = [item[2].get("text", "") for item in valid_items]
        embeddings = model.encode(texts)
        selected_indices = mmr(embeddings, prompt_embedding, top_n=k)

        results = []
        for sel_idx in selected_indices:
            idx, score, metadata = valid_items[sel_idx]
            results.append({
                "dataset": dataset_name,
                "score": score,
                "data": metadata
            })

        return results

    except Exception as e:
        logger.error(f"Error searching dataset '{dataset_name}': {e}")
        return []

def balanced_search(prompt_embedding, mode, total_results=TOP_K):
    """Perform balanced search across all datasets"""
    available_datasets = []
    for dataset_name, data in index_data.items():
        mapping = data["mapping"]
        if mode in mapping and len(mapping[mode]["indices"]) > 0:
            available_datasets.append(dataset_name)
    
    if not available_datasets:
        return []
    
    datasets_count = len(available_datasets)
    base_per_dataset = max(MIN_RESULTS_PER_DATASET, total_results // datasets_count)
    
    all_results = []
    dataset_results = {}
    
    for dataset_name in available_datasets:
        data = index_data[dataset_name]
        results = search_dataset(dataset_name, data, prompt_embedding, mode, base_per_dataset)
        dataset_results[dataset_name] = results
        all_results.extend(results)
    
    current_total = len(all_results)
    if current_total < total_results:
        remaining_needed = total_results - current_total
        dataset_result_counts = [(len(dataset_results[name]), name) for name in available_datasets]
        dataset_result_counts.sort(reverse=True)
        
        for _, dataset_name in dataset_result_counts:
            if remaining_needed <= 0:
                break
                
            current_count = len(dataset_results[dataset_name])
            can_add = min(remaining_needed, MAX_RESULTS_PER_DATASET - current_count, 10)
            
            if can_add > 0:
                data = index_data[dataset_name]
                additional_results = search_dataset(
                    dataset_name, data, prompt_embedding, mode, 
                    current_count + can_add
                )[current_count:]
                
                all_results.extend(additional_results)
                remaining_needed -= len(additional_results)
    
    all_results = sorted(all_results, key=lambda x: x["score"])[:total_results]
    
    result_distribution = {}
    for result in all_results:
        dataset = result["dataset"]
        result_distribution[dataset] = result_distribution.get(dataset, 0) + 1
    
    logger.info(f"Result distribution: {result_distribution}")
    return all_results

import tiktoken  # Or use the appropriate tokenizer if Groq has their own

GROQ_MODEL_NAME = "llama3-70b-8192"
MODEL_TOKEN_LIMIT = 8192

def count_tokens(text, model=GROQ_MODEL_NAME):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def generate_completion(groq_client, system_prompt, user_message, stream=False, temperature=0.3):
    # Token budget calculation
    input_tokens = count_tokens(system_prompt) + count_tokens(user_message)
    buffer_tokens = 50  # buffer for safety margin
    max_allowed_output = MODEL_TOKEN_LIMIT - input_tokens - buffer_tokens
    max_tokens = min(3000, max_allowed_output)

    if max_tokens <= 0:
        raise ValueError("❌ Input too long — reduce prompt length or context.")

    print(f"🧮 Input tokens: {input_tokens} | Max output tokens: {max_tokens} | Streaming: {stream}")

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    if stream:
        stream_resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=True
        )

        print("📡 Streaming response:")
        full_response = ""
        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_response += delta

        print("\n✅ Done streaming.")
        return full_response

    else:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=False
        )

        output_text = completion.choices[0].message.content
        finish_reason = completion.choices[0].finish_reason

        print(f"📏 Finish Reason: {finish_reason} | Output length: {len(output_text.split())} words")
        return output_text


def get_dataset_prompt_template(dataset_name: str) -> Dict[str, str]:
    """Get system and user prompt templates for each dataset type"""
    templates = {
        "courses": {
            "system": """You are a content enhancement specialist. Analyze and improve course data in batch format. 
Your task is to enhance multiple course entries at once, filling empty or missing fields with relevant, accurate information.

Guidelines:
- Keep existing good content unchanged
- Make descriptions concise but informative (max 150 words each)
- Ensure all fields are meaningful and relevant
- Don't repeat content across different courses
- Be specific and actionable""",
            
            "user": """Query: "{query}"
Mode: {mode}

Please enhance these course entries with improved content:

{items_json}

For each course, ensure these fields are properly filled:
- page_title: Keep original or improve if needed
- page_description: Clear, engaging description highlighting key benefits
- fee_detail: If empty, provide typical fee structure or "Contact for pricing"
- learning_term_details: Detailed syllabus/curriculum outline
- course_highlight: Key features, level, format, certification info
- job_details: Relevant career opportunities and job roles

Return a JSON array with the same structure but enhanced content. Ensure each course is unique and relevant to the query."""
        },
        
        "jobs": {
            "system": """You are a job content enhancement specialist. Analyze and improve job listing data in batch format.
Your task is to enhance multiple job entries at once, filling empty or missing fields with relevant, accurate information.

Guidelines:
- Keep existing good content unchanged
- Make descriptions comprehensive but concise (max 150 words each)
- Ensure all fields are meaningful and job-relevant
- Don't repeat content across different jobs
- Be specific about requirements and benefits""",
            
            "user": """Query: "{query}"
Mode: {mode}

Please enhance these job entries with improved content:

{items_json}

For each job, ensure these fields are properly filled:
- title: Keep original job title
- description: Comprehensive job description with responsibilities
- company: Company name if available
- location: Job location
- salary: Salary range if available or "Competitive salary"
- requirements: Key skills and qualifications needed
- benefits: Employee benefits and perks

Return a JSON array with the same structure but enhanced content. Ensure each job is unique and relevant to the query."""
        },
        
        "certificates": {
            "system": """You are a certification content enhancement specialist. Analyze and improve certificate data in batch format.
Your task is to enhance multiple certificate entries at once, filling empty or missing fields with relevant, accurate information.

Guidelines:
- Keep existing good content unchanged
- Make descriptions clear and value-focused (max 150 words each)
- Ensure all fields are meaningful and certification-relevant
- Don't repeat content across different certificates
- Be specific about career benefits""",
            
            "user": """Query: "{query}"
Mode: {mode}

Please enhance these certificate entries with improved content:

{items_json}

For each certificate, ensure these fields are properly filled:
- page_title: Keep original certification name
- page_description: Clear description of certification value
- fee_detail: Fee structure or "Free" if applicable
- learning_term_details: What you'll learn and skills gained
- course_highlight: Duration, level, format, accreditation
- job_details: Career opportunities this certification opens

Return a JSON array with the same structure but enhanced content. Ensure each certificate is unique and relevant to the query."""
        },
        
        "ebooks": {
            "system": """You are an ebook content enhancement specialist. Analyze and improve ebook data in batch format.
Your task is to enhance multiple ebook entries at once, filling empty or missing fields with relevant, accurate information.

Guidelines:
- Keep existing good content unchanged
- Make descriptions engaging and informative (max 150 words each)
- Ensure all fields are meaningful and book-relevant
- Don't repeat content across different ebooks
- Be specific about content and benefits""",
            
            "user": """Query: "{query}"
Mode: {mode}

Please enhance these ebook entries with improved content:

{items_json}

For each ebook, ensure these fields are properly filled:
- title: Keep original title
- description: Engaging description of content and benefits
- author: Author name if not present
- pages: Approximate page count if not present
- format: File format (PDF, EPUB, etc.)
- topics: Key topics covered
- level: Beginner/Intermediate/Advanced

Return a JSON array with the same structure but enhanced content. Ensure each ebook is unique and relevant to the query."""
        }
    }
    
    return templates.get(dataset_name, templates["courses"])

async def enhance_dataset_batch(dataset_name: str, items: List[Dict], query: str, mode: str) -> List[Dict]:
    """Enhance a batch of items from a single dataset with one LLM call"""
    if not groq_client or not items:
        return items
    
    try:
        # Get prompt template for this dataset
        template = get_dataset_prompt_template(dataset_name)
        
        # Prepare the batch data
        items_json = json.dumps(items, indent=2)
        
        system_prompt = template["system"]
        user_message = template["user"].format(
            query=query,
            mode=mode,
            items_json=items_json
        )
        
        # Make single LLM call for all items in this dataset
        # completion = groq_client.chat.completions.create(
        #     model="llama3-70b-8192",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_message}
        #     ],
        #     temperature=0.3,
        #     max_tokens=3000,  # Increased for batch processing
        #     top_p=1,
        #     stream=False
        # )

        completion = generate_completion(
            groq_client=groq_client,
            system_prompt=system_prompt,
            user_message=user_message,
            stream=True
        )
        
        # Parse the enhanced content
        enhanced_content = completion.choices[0].message.content.strip()
        
        # Extract JSON array from the response
        json_match = re.search(r'\[.*\]', enhanced_content, re.DOTALL)
        if json_match:
            try:
                enhanced_items = json.loads(json_match.group())
                
                # Validate that we got the expected number of items
                if isinstance(enhanced_items, list) and len(enhanced_items) > 0:
                    # Ensure we don't return more items than we sent
                    return enhanced_items[:len(items)]
                else:
                    logger.warning(f"LLM returned unexpected format for {dataset_name}")
                    return items
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing enhanced JSON for {dataset_name}: {e}")
                return items
        else:
            logger.warning(f"No JSON array found in LLM response for {dataset_name}")
            return items
            
    except Exception as e:
        logger.error(f"Error enhancing batch for {dataset_name}: {e}")
        return items

async def enhance_content_with_llm(query: str, raw_results: List[Dict], mode: str):
    """True single API call batch processing - all datasets in one request"""
    if not groq_client:
        return format_raw_results(raw_results)
    
    try:
        # Group results by dataset
        grouped_data = {}
        for result in raw_results:
            dataset = result["dataset"]
            if dataset not in grouped_data:
                grouped_data[dataset] = []
            grouped_data[dataset].append(result["data"])

        # Limit items per dataset to avoid token limits
        for dataset_name in grouped_data:
            grouped_data[dataset_name] = grouped_data[dataset_name][:8]

        # Create single comprehensive prompt for ALL datasets
        system_prompt = f"""You are a content enhancement specialist. Your task is to analyze and improve data across multiple datasets based on the search query: "{query}" for {mode} mode.

Instructions:
1. Fill empty or missing fields with relevant, accurate information
2. Keep existing content unchanged if it's already good
3. Make descriptions concise but informative (max 150 words each)
4. Ensure all fields are meaningful and relevant to the query
5. Don't repeat the same content across different items
6. Be specific and actionable
7. Maintain the exact same JSON structure for each dataset

Dataset-specific requirements:
- courses: page_title, page_description, fee_detail, learning_term_details, course_highlight, job_details
- jobs: title, description, company, location, salary, requirements, benefits
- certificates: page_title, page_description, fee_detail, learning_term_details, course_highlight, job_details
- ebooks: title, description, author, pages, format, topics, level

Return the enhanced data in the EXACT same structure as provided, grouped by dataset."""

        # Prepare the comprehensive batch data
        batch_data = json.dumps(grouped_data, indent=2)
        
        user_message = f"""Query: "{query}"
Mode: {mode}

Please enhance all the following data across multiple datasets:

{batch_data}

Return the enhanced data in the exact same JSON structure, grouped by dataset name. Ensure each item within each dataset is unique and relevant to the query."""

        # Make SINGLE API call for ALL datasets
        logger.info("Making single API call for all datasets")
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=3000,  # Increased for all datasets
            top_p=1,
            stream=False
        )
        
        # Parse the enhanced content
        enhanced_content = completion.choices[0].message.content.strip()
        
        # Extract JSON object from the response
        json_match = re.search(r'\{.*\}', enhanced_content, re.DOTALL)
        if json_match:
            try:
                enhanced_all_data = json.loads(json_match.group())
                
                # Validate that we got the expected structure
                if isinstance(enhanced_all_data, dict):
                    enhanced_results = {}
                    
                    # Process each dataset from the single response
                    for dataset_name, items in enhanced_all_data.items():
                        if dataset_name in grouped_data and isinstance(items, list):
                            # Remove duplicates and ensure uniqueness
                            unique_items = []
                            seen_titles = set()
                            
                            for item in items:
                                title = item.get('title', item.get('page_title', ''))
                                if title and title not in seen_titles:
                                    seen_titles.add(title)
                                    unique_items.append(item)
                            
                            enhanced_results[dataset_name] = unique_items
                        else:
                            # Fallback to original data for this dataset
                            enhanced_results[dataset_name] = grouped_data.get(dataset_name, [])
                    
                    # Ensure all original datasets are included
                    for dataset_name in grouped_data:
                        if dataset_name not in enhanced_results:
                            enhanced_results[dataset_name] = grouped_data[dataset_name]
                    
                    logger.info("Successfully enhanced all datasets with single API call")
                    return enhanced_results
                else:
                    logger.warning("LLM returned unexpected format, using original data")
                    return format_raw_results(raw_results)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing enhanced JSON: {e}")
                return format_raw_results(raw_results)
        else:
            logger.warning("No JSON object found in LLM response")
            return format_raw_results(raw_results)
            
    except Exception as e:
        logger.error(f"Error in single-call content enhancement: {e}")
        return format_raw_results(raw_results)

def format_raw_results(raw_results: List[Dict]):
    """Format raw results without LLM enhancement"""
    grouped_data = {}
    for result in raw_results:
        dataset = result["dataset"]
        if dataset not in grouped_data:
            grouped_data[dataset] = []
        grouped_data[dataset].append(result["data"])
    
    # Remove duplicates and limit results
    for dataset_name in grouped_data:
        unique_items = []
        seen_titles = set()
        
        for item in grouped_data[dataset_name][:8]:
            title = item.get('title', item.get('page_title', ''))
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_items.append(item)
        
        grouped_data[dataset_name] = unique_items
    
    return grouped_data

@app.get("/")
def root():
    return {
        "message": "Smart Vector Search API with Content Enhancement",
        "version": "2.0.0",
        "available_datasets": list(index_data.keys()),
        "total_datasets": len(index_data),
        "llm_enabled": groq_client is not None,
        "features": [
            "Vector similarity search",
            "Multi-dataset balanced search", 
            "Single API call content enhancement",
            "Duplicate removal",
            "Structured data formatting"
        ]
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
async def search(
    prompt: str = Query(..., description="Search prompt", min_length=1),
    mode: Literal["student", "professional"] = Query(..., description="Entry type"),
    enhance_content: bool = Query(True, description="Use LLM to enhance and fill missing content"),
    max_results: int = Query(20, description="Maximum number of results", ge=5, le=50)
):
    """Enhanced search endpoint with optimized batch content enhancement"""
    try:
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Search prompt cannot be empty")
        
        try:
            prompt_embedding = model.encode([prompt.strip()])[0].astype(np.float32)
            
            if np.any(np.isnan(prompt_embedding)) or np.any(np.isinf(prompt_embedding)):
                raise HTTPException(status_code=500, detail="Generated embedding contains invalid values")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(status_code=500, detail="Error processing search prompt")

        # Get raw search results
        raw_results = balanced_search(prompt_embedding, mode, max_results)
        
        if not raw_results:
            return {
                "query": prompt.strip(),
                "mode": mode,
                "total_results": 0,
                "message": "No relevant results found for your query.",
                "results": {}
            }

        # Enhance content if requested and LLM is available (now with single API call)
        if enhance_content and groq_client:
            enhanced_results = await enhance_content_with_llm(prompt.strip(), raw_results, mode)
        else:
            enhanced_results = format_raw_results(raw_results)

        return {
            "query": prompt.strip(),
            "mode": mode,
            "total_results": sum(len(items) for items in enhanced_results.values()),
            "datasets_found": len(enhanced_results),
            "content_enhanced": enhance_content and groq_client is not None,
            "enhancement_method": "single_api_call" if enhance_content and groq_client else "none",
            "timestamp": datetime.now().isoformat(),
            "results": enhanced_results
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
        "embedding_dimension": model.get_sentence_embedding_dimension() if model else None,
        "llm_enabled": groq_client is not None,
        "optimization": "single_api_call_enabled",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)