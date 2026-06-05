from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    try:
        pass

    except Exception as e:
        pass