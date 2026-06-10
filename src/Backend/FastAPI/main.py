from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from src.Backend.FastAPI.routes import router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="FLOW API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = Path(__file__).parent.parent.parent / "Frontend"

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/")
def serve_ui():
    return FileResponse(FRONTEND / "front.html")

app.include_router(router)


@app.get("/gen")
async def root():
    return {"status": "running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}