from fastapi import FastAPI, status

from backend.api.ingest import router as ingest_router
from backend.api.retrieval import router as retrieval_router

app = FastAPI()
app.include_router(ingest_router)
app.include_router(retrieval_router)


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {"message": "API is running. Go to /docs or /health"}


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}
