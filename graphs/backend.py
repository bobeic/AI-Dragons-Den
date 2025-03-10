import os
import asyncio
import random
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from IPython.display import Image, display

# Load API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Initialize models
entrepreneur_llm = ChatMistralAI(model="mistral-small-latest", temperature=0, max_retries=2, streaming=True)
tech_dragon_llm = ChatMistralAI(model="mistral-small-latest", temperature=0, max_retries=2, streaming=True)
health_dragon_llm = ChatMistralAI(model="mistral-small-latest", temperature=0, max_retries=2, streaming=True)

# Define prompts
entrepreneur_prompt = PromptTemplate.from_template("""
You are an ambitious entrepreneur pitching a startup idea on Dragon's Den.

Your task:
1. Identify a major problem in the {industry} industry.
2. Propose an innovative business idea that solves this problem.
3. Explain your unique selling point (USP) – what makes your idea different?
4. Describe your business model – how will you generate revenue?

Be compelling and persuasive. Keep your pitch under 100 words.

Your pitch:""")

tech_dragon_prompt = PromptTemplate.from_template("""
You are a seasoned investor on Dragon's Den with extensive expertise in technology and finance. 

Your task:
1. Analyze the business idea: {pitch} with a focus on its technological feasibility and financial viability.
2. Identify the strengths – what makes it promising?
3. Highlight potential risks or weaknesses, particularly in terms of tech scalability and financial sustainability.
4. Assess market potential – is there demand? Who are the competitors?
5. Suggest improvements or alternative business strategies, especially from a tech and finance perspective.

Provide your analysis in a professional but engaging way, like a real Dragon's Den judge. Keep your response under 200 words.

Your response:""")

health_dragon_prompt = PromptTemplate.from_template("""
You are a seasoned investor on Dragon's Den with deep expertise in healthcare and medical innovations. 

Your task:
1. Analyze the business idea: {pitch} with a focus on its applicability in healthcare and potential medical impact.
2. Identify the strengths – what makes it promising?
3. Highlight potential risks or weaknesses, especially in regulatory compliance, patient safety, and healthcare adoption.
4. Assess market potential – is there demand? Who are the competitors?
5. Suggest improvements or alternative business strategies, particularly from a healthcare industry perspective.

Provide your analysis in a professional but engaging way, like a real Dragon's Den judge. Keep your response under 200 words.

Your response:""")

response_prompt = PromptTemplate.from_template("""
You are the entrepreneur responding to investor feedback. Address the points raised in their evaluation and explain how your business will overcome challenges. Keep your response under 100 words.

Investor's feedback: {evaluation}

Your response:""")

# Generate AI pitch
def generate_demo_pitch(industry: str):
    entrepreneur_chain = entrepreneur_prompt | entrepreneur_llm
    response = entrepreneur_chain.invoke({"industry": industry})
    return response

template_industry = "fintech"
pitch = generate_demo_pitch(template_industry)
print(pitch.content)

# Define state
class ConversationState(MessagesState):
    industry: str = template_industry
    pitch: str
    summary: str

# Define investor evaluations
def evaluate_pitch(state: ConversationState, dragon_llm, dragon_prompt):
    pitch = state["messages"][-1].content
    dragon_chain = dragon_prompt | dragon_llm
    response = dragon_chain.invoke({"pitch": pitch})
    return {"messages": response}

def evaluate_pitch_tech_dragon(state: ConversationState):
    return evaluate_pitch(state, tech_dragon_llm, tech_dragon_prompt)

def evaluate_pitch_health_dragon(state: ConversationState):
    return evaluate_pitch(state, health_dragon_llm, health_dragon_prompt)

def entrepreneur_response(state: ConversationState):
    evaluations = "\n\n".join(msg.content for msg in state["messages"][-2:])  # Combine both dragons' feedback
    response_chain = response_prompt | entrepreneur_llm
    response = response_chain.invoke({"evaluation": evaluations})
    return {"messages": response}

def pick_random_dragon(state):
    
    # user_input = state['graph_state'] 
    
    if random.random() < 0.5:

        return "tech_dragon"
    
    return "health_dragon"


# Define workflow
builder = StateGraph(ConversationState)

# dragon_nodes = ["tech_dragon", "health_dragon"]
# random.shuffle(dragon_nodes)  # Shuffle order dynamically

builder.add_node("tech_dragon", evaluate_pitch_tech_dragon)
builder.add_node("health_dragon", evaluate_pitch_health_dragon)
builder.add_node("entrepreneur_response", entrepreneur_response)

# builder.add_edge(START, dragon_nodes[0])
builder.add_conditional_edges(START, pick_random_dragon)
# builder.add_edge(dragon_nodes[0], dragon_nodes[1])
builder.add_edge("tech_dragon", "entrepreneur_response")
builder.add_edge("health_dragon", "entrepreneur_response")
builder.add_edge("entrepreneur_response", END)

graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

input_message = HumanMessage(content=pitch.content)
for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
