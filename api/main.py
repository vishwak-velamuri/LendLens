from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.analysis import router as analysis_router
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(
    title="CASCA Loan Analyzer",
    description="Upload a bank-statement PDF + loan terms, get back metrics & an approve/deny decision.",
    version="0.1",
)

# Allow your React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React runs here
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

# Mount router
app.include_router(analysis_router, prefix="/api", tags=["analysis"])

# Mount static files
HERE = os.path.dirname(__file__)
build_dir = os.path.abspath(os.path.join(HERE, "../frontend/build"))

app.mount("/", StaticFiles(directory=build_dir, html=True), name="static")