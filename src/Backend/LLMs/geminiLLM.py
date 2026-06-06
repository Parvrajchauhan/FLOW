import os
from langchain_google_genai import GoogleGenerativeAI

import os
from dotenv import load_dotenv
load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")



class GeminiLLM:
    def __init__(self):
        pass
        
    def get_llm(self):
        try:
            if api_key=='':
                pass
                
            llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)
      
        except Exception as e:
            pass
        return llm