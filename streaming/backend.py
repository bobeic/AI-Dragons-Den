import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate

# Load API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize models
entrepreneur_llm = ChatMistralAI(
    model="mistral-small-latest", temperature=0, max_retries=2, streaming=True
)

dragon_llm = ChatMistralAI(
    model="mistral-small-latest", temperature=0, max_retries=2, streaming=True
)

# Define prompts
entrepreneur_prompt = PromptTemplate.from_template(
    """You are an ambitious entrepreneur pitching a startup idea on Dragon's Den.
    
    Your task:
    1. Identify a major problem in the {industry} industry.
    2. Propose an innovative business idea that solves this problem.
    3. Explain your unique selling point (USP) – what makes your idea different?
    4. Describe your business model – how will you generate revenue?
    
    Be compelling and persuasive. Keep your pitch under 100 words.

    Your pitch:"""
)

dragon_prompt = PromptTemplate.from_template(
    """You are a seasoned investor on Dragon's Den, evaluating a startup pitch. 
    
    Your task:
    1. Analyze the business idea: {idea}
    2. Identify the strengths – what makes it promising?
    3. Highlight potential risks or weaknesses.
    4. Assess market potential – is there demand? Who are the competitors?
    5. Suggest improvements or alternative business strategies.

    Provide your analysis in a professional but engaging way, like a real Dragon's Den judge. Keep your response under 200 words.

    Your response:"""
)


class QueryRequest(BaseModel):
    industry: str


class EvaluationRequest(BaseModel):
    idea: str


@app.post("/generate_pitch")
async def generate_pitch(request: QueryRequest):
    """Asynchronously streams the entrepreneur's pitch."""
    entrepreneur_chain = entrepreneur_prompt | entrepreneur_llm

    async def token_generator():
        async for chunk in entrepreneur_chain.astream({"industry": request.industry}):
            yield chunk.content  # Stream token content
            await asyncio.sleep(0)

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.post("/evaluate_pitch")
async def evaluate_pitch(request: EvaluationRequest):
    """Asynchronously streams the investor's evaluation."""
    dragon_chain = dragon_prompt | dragon_llm

    async def token_generator():
        async for chunk in dragon_chain.astream({"idea": request.idea}):
            yield chunk.content  # Stream token content
            await asyncio.sleep(0)

    return StreamingResponse(token_generator(), media_type="text/plain")
