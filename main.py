# main.py
from fastapi import FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import logging

# Import functions from modules
from signup import UserSignup, UserLogin, init_database, create_user, authenticate_user
from llm_query import search_logic, get_datasets_info, health_check_logic, root_info
from external_jobs import extract_skills_from_live_jobs
from fastapi import FastAPI, Query, HTTPException
from typing import Literal
import logging, uvicorn
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI app
app = FastAPI(title="AI Navigator - Clean Architecture", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ AUTH ROUTES ============
@app.post("/signup")
async def signup_endpoint(user: UserSignup):
    """User registration endpoint"""
    return await create_user(user)

@app.post("/login")
async def login_endpoint(user: UserLogin):
    """User login endpoint"""
    return await authenticate_user(user)


@app.get("/")
def root():
    return root_info()

@app.get("/datasets")
def get_datasets():
    return get_datasets_info()

@app.get("/search")
async def search(
    prompt: str = Query(..., description="Search prompt", min_length=1),
    mode: Literal["student", "professional"] = Query(..., description="Entry type"),
    enhance_content: bool = Query(True, description="Use LLM to enhance and fill missing content"),
    max_results: int = Query(20, description="Maximum number of results", ge=5, le=50)
):
    return await search_logic(prompt, mode, enhance_content, max_results)

@app.get("/health")
def health_check():
    return health_check_logic()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)