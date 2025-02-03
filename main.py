import os
import time
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

logging.basicConfig(level=logging.INFO)

# model_name = "deepseek-ai/DeepSeek-V3"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

dragon_llm = HuggingFaceEndpoint(
    repo_id=model_name,
    task="text-generation",
    temperature=0.5,
    huggingfacehub_api_token=api_key,
)

entrepreneur_llm = HuggingFaceEndpoint(
    repo_id=model_name,
    task="text-generation",
    temperature=0.5,
    huggingfacehub_api_token=api_key,
)

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
    
    Provide your analysis in a professional but engaging way, like a real Dragon's Den judge.
    
    Your response:"""
)

entrepreneur_chain = entrepreneur_prompt | entrepreneur_llm
dragon_chain = dragon_prompt | dragon_llm


industry = "fitness technology"

logging.info(f"Generating entrepreneur pitch for industry: {industry}")

start_time = time.time()
idea = entrepreneur_chain.invoke({"industry": industry})  # Entrepreneur pitches
end_time = time.time()

logging.info(f"Entrepreneur LLM execution time: {end_time - start_time:.2f} seconds")
logging.info(f"Entrepreneur Pitch: {idea}")

start_time = time.time()
response = dragon_chain.invoke({"idea": idea})  # Judge evaluates
end_time = time.time()

logging.info(f"Dragon LLM execution time: {end_time - start_time:.2f} seconds")
logging.info(f"Dragon Response: {response}")

# Print results
print("\n*** Entrepreneur's Pitch ***")
print(idea)
print("\n*** Dragon's Response ***")
print(response)