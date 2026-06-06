import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()


def get_llm() -> GoogleGenerativeAI:
    """Create and return a Gemini LLM instance"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        return GoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api_key)

    except ValueError:
        raise

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini LLM: {str(e)}") from e