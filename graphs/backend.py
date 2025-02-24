import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
# from langgraph.checkpoint import MemorySaver
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

from IPython.display import Image, display

# Load API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Initialize FastAPI
# app = FastAPI()

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

# Define state for conversation tracking
# class ConversationState(BaseModel):
#     industry: str = ""
#     pitch: str = ""
#     evaluation: str = ""

class State(MessagesState):
    summary: str

# Initialize memory for storing conversation state
# memory = MemorySaver()

# Define LangGraph workflow
# graph = StateGraph(ConversationState)
graph = StateGraph(State)

# Step 1: Generate entrepreneur's pitch
def generate_pitch(state: State):
    entrepreneur_chain = entrepreneur_prompt | entrepreneur_llm
    response = entrepreneur_chain.invoke({"industry": state.industry})
    state.pitch = response.content
    return state

# Step 2: Investor evaluates pitch
def evaluate_pitch(state: State):
    dragon_chain = dragon_prompt | dragon_llm
    response = dragon_chain.invoke({"idea": state.pitch})
    state.evaluation = response.content
    return state

# Add nodes to the graph
graph.add_node("entrepreneur", RunnableLambda(generate_pitch))
graph.add_node("investor", RunnableLambda(evaluate_pitch))

# Define edges (execution order)
graph.add_edge(START, "entrepreneur")
# graph.set_entry_point("entrepreneur")
graph.add_edge("entrepreneur", "investor")  # Entrepreneur -> Investor

# Compile the graph
# app_graph = graph.compile(checkpointer=memory)
app_graph = graph.compile()
display(Image(app_graph.get_graph().draw_mermaid_png()))

def simulate_llm_conversation():
    entrepreneur_pitch = "Hello Dragons, I am here to present my innovative idea that will change the world!"
    evaluator_feedback = "Thank you for your pitch. While your idea is interesting, I have some concerns regarding its feasibility."
    entrepreneur_reaction = "I appreciate your feedback and understand your concerns. Let me explain how I plan to address them."
    
    return {
        'pitch': entrepreneur_pitch,
        'feedback': evaluator_feedback,
        'reaction': entrepreneur_reaction
    }

# FastAPI Endpoint for running the LangGraph workflow
# @app.post("/start_conversation")
# async def start_conversation(request: QueryRequest):
#     """Starts the conversation flow and streams the responses."""
    
#     async def token_generator():
#         async for event in app_graph.astream_events(
#             {"industry": request.industry},  # Start with industry input
#             send_events=True
#         ):
#             if "entrepreneur" in event:
#                 yield f"Entrepreneur: {event['entrepreneur']['pitch']}\n"
#             if "investor" in event:
#                 yield f"Investor: {event['investor']['evaluation']}\n"
#             await asyncio.sleep(0)

#     return StreamingResponse(token_generator(), media_type="text/plain")
