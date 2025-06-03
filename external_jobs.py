import re
import json
import os
import pickle
import faiss
from fastapi import Query, HTTPException
from llm_query import groq_client, generate_completion
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables for embeddings data
embeddings_data = {}

def load_embeddings_data():
    """Load embeddings data from static/gap_embeddings directory"""
    global embeddings_data
    
    embeddings_dir = "static/gap_embeddings"
    
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    
    try:
        # Load jobs data with actual filenames
        jobs_index_path = os.path.join(embeddings_dir, "jobs.index")
        jobs_metadata_path = os.path.join(embeddings_dir, "jobs.pkl")
        
        if os.path.exists(jobs_index_path) and os.path.exists(jobs_metadata_path):
            # Load FAISS index
            jobs_index = faiss.read_index(jobs_index_path)
            
            # Load metadata
            with open(jobs_metadata_path, 'rb') as f:
                jobs_metadata = pickle.load(f)
            
            embeddings_data["jobs"] = {
                "index": jobs_index,
                "metadata": jobs_metadata
            }
            
        # Load additional embedding files if they exist (with .index and .pkl pattern)
        for filename in os.listdir(embeddings_dir):
            if filename.endswith('.index'):
                category = filename.replace('.index', '')
                if category != 'jobs':  # Already loaded jobs
                    index_path = os.path.join(embeddings_dir, filename)
                    metadata_path = os.path.join(embeddings_dir, f"{category}.pkl")
                    
                    if os.path.exists(metadata_path):
                        try:
                            index = faiss.read_index(index_path)
                            with open(metadata_path, 'rb') as f:
                                metadata = pickle.load(f)
                            
                            embeddings_data[category] = {
                                "index": index,
                                "metadata": metadata
                            }
                        except Exception as e:
                            pass
                            
    except Exception as e:
        raise

def extract_json_from_response(response_text):
    # Extract JSON inside triple backticks or first JSON array/object in text
    json_match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"```(.*?)```", response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\[.*\]|\{.*\})", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return None
    return None

def generate_learning_path_prompt(target_role, current_level):
    return f"""
You are a career coach helping users progress from their current skill level to their target job role.

User Goal: "{target_role}"
Current Level: "{current_level}"

Create a detailed, step-by-step learning path with specific, actionable steps. Each step should build upon the previous one.

IMPORTANT: You must respond with ONLY a valid JSON array. No other text before or after.

Format example:
[
  {{
    "step_number": 1,
    "duration": "4-6 weeks",
    "skills": ["Programming Fundamentals", "Version Control", "Basic Data Structures"],
    "activities": ["Complete Python/Java basics course", "Set up GitHub account and practice Git", "Solve 50 basic coding problems on LeetCode"]
  }},
  {{
    "step_number": 2,
    "duration": "6-8 weeks", 
    "skills": ["Object-Oriented Programming", "Database Basics", "Web Development"],
    "activities": ["Build 3 small projects using OOP principles", "Learn SQL and database design", "Create a simple web application"]
  }}
]

Provide 4-6 detailed steps specifically for becoming a {target_role} from {current_level} level.
"""

def generate_skill_enhancement_prompt(target_role):
    return f"""
You are an expert career advisor specializing in {target_role} positions.

IMPORTANT: Respond with ONLY a valid JSON array. No explanatory text before or after.

Provide the top 8-10 most critical skills for a {target_role} role:

[
  {{
    "skill_name": "Advanced Programming",
    "description": "Expert-level coding in multiple languages with focus on clean, scalable code",
    "learning_tips": "Master design patterns, contribute to open source, practice system design"
  }},
  {{
    "skill_name": "System Architecture",
    "description": "Ability to design scalable, maintainable software systems",
    "learning_tips": "Study distributed systems, practice designing large-scale applications, learn microservices"
  }}
]

Focus on skills specific to {target_role} level responsibilities.
"""

def semantic_rank_skills(target_role, skills):
    # Filter valid skills
    valid_skills = [s for s in skills if all(k in s for k in ("skill_name", "description", "learning_tips"))]
    if not valid_skills:
        return []

    try:
        target_emb = model.encode([target_role])[0].reshape(1, -1)
        skill_texts = [f"{s['skill_name']}: {s['description']}" for s in valid_skills]
        skill_embs = model.encode(skill_texts)
        similarities = cosine_similarity(skill_embs, target_emb).flatten()
        ranked = sorted(zip(valid_skills, similarities), key=lambda x: x[1], reverse=True)
        return [skill for skill, score in ranked]
    except Exception as e:
        return valid_skills[:10]  # Return first 10 skills as fallback

def extract_skills_from_jobs(jobs):
    skills = []
    for job in jobs:
        skill_text = job.get("skills_required") or job.get("skills") or job.get("enhanced_skills")
        if skill_text:
            for s in skill_text.split(","):
                skill_name = s.strip()
                if skill_name:
                    skills.append({
                        "skill_name": skill_name,
                        "description": f"Essential skill for {job.get('title', 'this role')}",
                        "learning_tips": f"Focus on practical application of {skill_name}"
                    })
    return skills

def create_fallback_job_result(query, target_job):
    """Create a fallback job result when no jobs are found"""
    return {
        "title": target_job,
        "enhanced_description": f"We couldn't find specific job postings for '{query}', but here's what a {target_job} role typically involves: This position requires strong technical skills, problem-solving abilities, and continuous learning mindset.",
        "enhanced_skills": "Problem-solving, Communication, Technical expertise, Adaptability, Continuous learning"
    }

def create_default_learning_path(target_job, current_level):
    """Create a default learning path when LLM fails"""
    return [
        {
            "step_number": 1,
            "duration": "1-2 months",
            "skills": ["Fundamentals", "Basic concepts"],
            "activities": [f"Learn the basics of {target_job}", "Take introductory courses", "Read industry blogs"]
        },
        {
            "step_number": 2,
            "duration": "2-3 months", 
            "skills": ["Practical skills", "Hands-on experience"],
            "activities": ["Build projects", "Practice with real-world scenarios", "Join online communities"]
        },
        {
            "step_number": 3,
            "duration": "3-4 months",
            "skills": ["Advanced topics", "Specialization"],
            "activities": ["Take advanced courses", "Work on complex projects", "Network with professionals"]
        }
    ]

def create_default_skills(target_job):
    """Create default skills when LLM fails"""
    return [
        {
            "skill_name": "Technical Expertise",
            "description": f"Core technical skills required for {target_job}",
            "learning_tips": "Practice regularly and build projects"
        },
        {
            "skill_name": "Problem Solving",
            "description": "Ability to analyze and solve complex problems",
            "learning_tips": "Work on coding challenges and real-world problems"
        },
        {
            "skill_name": "Communication",
            "description": "Effective written and verbal communication skills",
            "learning_tips": "Practice presenting ideas and writing documentation"
        }
    ]

def filter_jobs_by_relevance(results, target_job, query, top_k=1):
    """Filter and rank jobs by semantic similarity to target job"""
    if not results:
        return []
    
    try:
        # Create combined query for better matching
        search_text = f"{target_job} {query}"
        target_emb = model.encode([search_text])[0].reshape(1, -1)
        
        # Score each job based on title and description relevance
        job_scores = []
        for job in results:
            job_text = f"{job.get('title', '')} {job.get('description', '')}"
            job_emb = model.encode([job_text])[0].reshape(1, -1)
            similarity = cosine_similarity(job_emb, target_emb)[0][0]
            job_scores.append((job, similarity))
        
        # Sort by similarity and return top_k
        job_scores.sort(key=lambda x: x[1], reverse=True)
        filtered_jobs = [job for job, score in job_scores[:top_k]]
        
        return filtered_jobs
        
    except Exception as e:
        return results[:top_k]

def search_jobs_in_embeddings(query, top_k=20):
    """Search for jobs in the loaded embeddings"""
    if "jobs" not in embeddings_data:
        return []
    
    try:
        jobs_data = embeddings_data["jobs"]
        index = jobs_data["index"]
        metadata = jobs_data["metadata"]
        
        # Generate query embedding
        query_embedding = model.encode([query])[0].astype(np.float32)
        
        # Search in FAISS index
        search_k = min(top_k, index.ntotal)
        D, I = index.search(np.array([query_embedding]), search_k)
        
        # Extract results
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(metadata):
                job_data = metadata[idx]
                results.append(job_data)
        
        return results
        
    except Exception as e:
        return []

async def search_jobs_endpoint(
    query: str = Query(..., description="Search query"),
    target_job: str = Query(..., description="Target job role"),
    current_level: str = Query("Beginner", description="User's current skill level"),
    top_k: int = Query(1, ge=1, le=20, description="Number of job results to return")
):
    # Ensure embeddings are loaded
    if not embeddings_data:
        try:
            load_embeddings_data()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load embeddings: {str(e)}")
    
    if "jobs" not in embeddings_data:
        raise HTTPException(status_code=500, detail="Jobs embeddings not available")

    # Search jobs semantically by combining query and target_job for better results
    combined_query = f"{query} {target_job}"
    
    # Search with higher k initially to get more candidates for filtering
    search_k = min(20, embeddings_data["jobs"]["index"].ntotal)
    initial_results = search_jobs_in_embeddings(combined_query, search_k)
    
    # Filter jobs by relevance to target_job and get only top_k
    results = filter_jobs_by_relevance(initial_results, target_job, query, top_k)

    # Handle case when no results found
    if not results:
        cleaned_results = [create_fallback_job_result(query, target_job)]
        fallback_skills = create_default_skills(target_job)
    else:
        # Enhance job descriptions and skills
        job_entries = []
        for job in results:
            job_entries.append(
                f"Job Title: {job.get('title', 'N/A')}\n"
                f"Current Description: {job.get('description', 'N/A')}\n"
                f"Current Skills: {job.get('skills_required', job.get('skills', 'N/A'))}\n"
                "Please enhance the description and skills to be more attractive and relevant to the job title."
            )
        jobs_text = "\n\n".join(job_entries)

        enhancement_prompt = f"""You are a professional career content writer.

User Query: "{query}"
Target Job Role: "{target_job}"

Enhance the following job postings by rewriting the description and skills sections to be more engaging, clear, and tailored to the job title. Keep the meaning but improve attractiveness and relevance.

{jobs_text}

IMPORTANT: Return ONLY a valid JSON array with exactly {len(results)} objects. No other text.

[
  {{
    "title": "Enhanced Job Title",
    "enhanced_description": "Enhanced engaging description...",
    "enhanced_skills": "Enhanced relevant skills..."
  }}
]
"""

        try:
            llm_response = generate_completion(groq_client=groq_client, system_prompt="", user_message=enhancement_prompt)
            
            enhanced_jobs = extract_json_from_response(llm_response)
            if enhanced_jobs is None or not isinstance(enhanced_jobs, list):
                enhanced_jobs = []
        except Exception as e:
            enhanced_jobs = []

        # Merge enhanced fields back into results
        cleaned_results = []
        for i, job in enumerate(results):
            if i < len(enhanced_jobs) and isinstance(enhanced_jobs[i], dict):
                enhanced = enhanced_jobs[i]
                cleaned_results.append({
                    "title": job.get("title", "N/A"),
                    "enhanced_description": enhanced.get("enhanced_description", job.get("description", "")),
                    "enhanced_skills": enhanced.get("enhanced_skills", job.get("skills_required", job.get("skills", "")))
                })
            else:
                # Fallback to original data
                cleaned_results.append({
                    "title": job.get("title", "N/A"),
                    "enhanced_description": job.get("description", ""),
                    "enhanced_skills": job.get("skills_required", job.get("skills", ""))
                })

        # Extract skills from jobs for fallback
        fallback_skills = extract_skills_from_jobs(results)

    # Generate learning path with better error handling
    learning_path = []
    try:
        learning_path_prompt = generate_learning_path_prompt(target_job, current_level)
        learning_path_response = generate_completion(groq_client=groq_client, system_prompt="You are a career coach. Always respond with valid JSON only.", user_message=learning_path_prompt)
        
        learning_path = extract_json_from_response(learning_path_response)
        if learning_path is None or not isinstance(learning_path, list):
            learning_path = create_default_learning_path(target_job, current_level)
    except Exception as e:
        learning_path = create_default_learning_path(target_job, current_level)

    # Generate skill enhancement list with better error handling
    ranked_skills = []
    try:
        skill_prompt = generate_skill_enhancement_prompt(target_job)
        skill_response = generate_completion(groq_client=groq_client, system_prompt="You are a career expert. Always respond with valid JSON only.", user_message=skill_prompt)
        
        skills_list = extract_json_from_response(skill_response)
        
        if not skills_list or not isinstance(skills_list, list):
            # Use fallback skills from jobs or default skills
            skills_list = fallback_skills if fallback_skills else create_default_skills(target_job)
        
        if skills_list:
            ranked_skills = semantic_rank_skills(target_job, skills_list)
        
    except Exception as e:
        # Use fallback skills
        skills_list = fallback_skills if fallback_skills else create_default_skills(target_job)
        ranked_skills = semantic_rank_skills(target_job, skills_list)

    # Ensure we always return some results
    if not learning_path:
        learning_path = create_default_learning_path(target_job, current_level)
    
    if not ranked_skills:
        ranked_skills = create_default_skills(target_job)

    return {
        "query": query,
        "target_job": target_job,
        "job_results": cleaned_results[:1],
        "learning_path": learning_path,
        "ranked_skills": ranked_skills
    }

# Initialize embeddings on module load
try:
    load_embeddings_data()
except Exception as e:
    pass