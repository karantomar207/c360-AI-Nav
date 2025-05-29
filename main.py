from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from scripts.query_embeddings import search

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search")
async def search_route(prompt: str = Query(...), mode: str = Query("student")):
    # Call the search function from query_embeddings.py
    results = search(prompt, mode)
    return results
