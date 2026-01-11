import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file location.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Med Mind AI Backend (OpenAI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SELECTED_MODEL = "gpt-4.1-mini" 

APP_API_KEY = os.getenv("MM_API_KEY", "medmind123")

def verify_key(x_api_key: str):
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

def ai_reply(user_message: str) -> str:
    try:
        # OpenAI Chat Completion Call
        response = client.chat.completions.create(
            model=SELECTED_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are Med Mind AI, a specialized medical health assistant. "
                        "STRICT RULE: You only answer questions related to health, medicine, symptoms, and wellness. "
                        "If a user asks about anything else (politics, sports, entertainment, general logic, etc.), "
                        "politely respond: 'I am sorry, but I am only programmed to assist with health and medical-related concerns.' "
                        "Always provide safe, general info and do not provide a formal diagnosis."

                        "FORMATTING RULES:\n"
                        "1. Use **bold text** for key medical terms, symptoms, or urgent advice.\n"
                        "2. Use bullet points for lists of symptoms or recommendations.\n"
                        "3. Use a '### Summary' or '### Key Takeaway' section for long responses.\n"
                        "4. Keep sentences concise. Use clear structure to ensure high readability."
                    )
                },
                {"role": "user", "content": user_message}
            ],
            temperature=0.2, # Lower temperature makes the AI more focused and less creative
            max_tokens=500
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"OPENAI API ERROR: {str(e)}")
        return "I'm having trouble connecting to my AI brain. Please try again later."

class PredictRequest(BaseModel):
    age: int
    gender: str
    symptoms: List[str]
    bp: Optional[int] = None
    glucose: Optional[int] = None

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.get("/health")
def health(x_api_key: str = Header(...)):
    verify_key(x_api_key)
    return {"status": "ok"}

@app.post("/chat")
def chat(data: ChatRequest, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    reply = ai_reply(data.message)
    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)