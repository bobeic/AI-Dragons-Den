import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load API key from .env
load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize Hugging Face client
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(model=model_name, token=api_key)


# Request model
class QueryRequest(BaseModel):
    industry: str


@app.post("/generate_pitch")
def generate_pitch(request: QueryRequest):
    """Handles requests to generate an entrepreneur's pitch."""
    entrepreneur_prompt = f"""
    You are an ambitious entrepreneur pitching a startup idea on Dragon's Den.

    Your task:
    1. Identify a major problem in the {request.industry} industry.
    2. Propose an innovative business idea that solves this problem.
    3. Explain your unique selling point (USP) – what makes your idea different?
    4. Describe your business model – how will you generate revenue?

    Be compelling and persuasive. Keep your pitch under 100 words.

    Your pitch:
    """

    idea = client.chat_completion(
        messages=[{"role": "user", "content": entrepreneur_prompt}]
    )

    return {"idea": idea}


class EvaluationRequest(BaseModel):
    idea: str


@app.post("/evaluate_pitch")
def evaluate_pitch(request: EvaluationRequest):
    """Handles requests to evaluate a startup pitch."""
    dragon_prompt = f"""
    You are a seasoned investor on Dragon's Den, evaluating a startup pitch. 
    
    Your task:
    1. Analyze the business idea: {request.idea}
    2. Identify the strengths – what makes it promising?
    3. Highlight potential risks or weaknesses.
    4. Assess market potential – is there demand? Who are the competitors?
    5. Suggest improvements or alternative business strategies.

    Provide your analysis in a professional but engaging way, like a real Dragon's Den judge.

    Your response:
    """

    response = client.chat_completion(
        messages=[{"role": "user", "content": dragon_prompt}]
    )

    return {"evaluation": response}


