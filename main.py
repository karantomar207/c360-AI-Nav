from fastapi import FastAPI, Request
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
def search_route(query: str):
    # The search function now returns the properly formatted data structure
    results = search(query)
    return results  # No need to wrap in {"results": results} as the structure is already correct