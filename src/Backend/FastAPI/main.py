from fastapi import FastAPI
from api.route.routes import router

app = FastAPI(title="Flow", version="1.0.0")

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "ok"}