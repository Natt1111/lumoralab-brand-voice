from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes.ingest import router as ingest_router
from app.routes.generate import router as generate_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nLumoraLab Brand Voice Generator — starting up")
    print(f"  Collection : {settings.collection_name}")
    print(f"  Chroma dir : {settings.chroma_persist_dir}")
    print(f"  Anthropic  : {'[set]' if settings.anthropic_api_key else '[MISSING]'}")
    print(f"  Voyage AI  : {'[set]' if settings.voyage_api_key else '[MISSING]'}")
    print()
    yield


app = FastAPI(
    title="LumoraLab Brand Voice Generator",
    description="RAG-powered content generator trained on LumoraLab brand content",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, prefix="/api")
app.include_router(generate_router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "ok", "collection": settings.collection_name}


# Mount static files LAST — the catch-all "/" would swallow any routes defined after it
app.mount("/", StaticFiles(directory="static", html=True), name="static")
