from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

import warmup_router

app = FastAPI(title="BassTutor Warm-Up Service")

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(warmup_router.router)

@app.get("/health")
def health():
    return {"ok": True}
