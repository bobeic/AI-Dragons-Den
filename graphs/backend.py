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
# from langchain_community.chat_models import ChatMistralAI

from models import get_chat_model

# Entrepreneur model
entrepreneur_llm = get_chat_model(model_name="llama3-8b-8192", provider="groq")

# Dragons
dragon_names = ["peter_jones", "deborah_meaden", "touker_suleyman", "steven_bartlett", "sara_davies"]

dragon_llms = {name: get_chat_model(model_name="llama3-8b-8192", provider="groq") for name in dragon_names}


from langchain_huggingface import HuggingFaceEmbeddings
from rag import create_dragon_vector_store

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from dragon_bios import dragon_bios

# Create Peter Jones' vector store
dragon_vectorstores = {}

for dragon_name, bio_texts in dragon_bios.items():
    store = create_dragon_vector_store(dragon_name, bio_texts)
    dragon_vectorstores[dragon_name] = store

dragons = {}

for dragon_name in dragon_names:
    dragons[dragon_name] = {
        "llm": dragon_llms[dragon_name],
        "vectorstore": dragon_vectorstores[dragon_name],
        "focus": dragon_bios[dragon_name][0]["focus"], 
        "personality": dragon_bios[dragon_name][0]["personality"], 
    }

# Define prompts
from prompts.entrepreneur_prompts import entrepreneur_pitch_prompt, entrepreneur_response_prompt, entrepreneur_counter_offer_prompt
from prompts.dragon_prompts import dragon_initial_evaluation_prompt, dragon_offer_making_prompt, dragon_negotiation_prompt

class State(MessagesState):
    industry: str  # The industry of the pitch
    pitch: str     # The original business pitch
    summary: str = ""  # Optional: could be used to summarize conversation later
    current_dragon: str = ""  # Who is the active dragon
    current_offer: str = ""  # Dragon's offer (if any)
    entrepreneur_counter_offer: str = ""  # Entrepreneur's counter-offer (if any)


def dragon_evaluation_node(state: State):
    pitch = state["messages"][-1].content
    
    dragon_name = state["current_dragon"]
    dragon_info = dragons[dragon_name]
    dragon_llm = dragon_info["llm"]
    evaluation_prompt = dragon_initial_evaluation_prompt

    # Retrieve dragon's focus and personality (optional)
    focus = dragon_info.get("focus", "general business")
    personality = dragon_info.get("personality", "professional")
    
    # Build prompt inputs
    input_data = {
        "dragon_name": dragon_name.replace("_", " ").title(),
        "focus": focus,
        "personality": personality,
        "pitch": pitch,
    }
    
    evaluation_chain = evaluation_prompt | dragon_llm
    response = evaluation_chain.invoke(input_data)

    return {"messages": [response]}

def entrepreneur_response_node(state: State):
    evaluation = state["messages"][-1].content
    
    # Entrepreneur model
    entrepreneur_chain = entrepreneur_response_prompt | entrepreneur_llm
    response = entrepreneur_chain.invoke({"evaluation": evaluation})

    return {"messages": [response]}

def dragon_offer_node(state: State):
    entrepreneur_reply = state["messages"][-1].content
    
    dragon_name = state["current_dragon"]
    dragon_info = dragons[dragon_name]
    dragon_llm = dragon_info["llm"]
    offer_prompt = dragon_offer_making_prompt

    focus = dragon_info.get("focus", "general business")
    personality = dragon_info.get("personality", "professional")
    pitch = state["pitch"]
    
    input_data = {
        "dragon_name": dragon_name.replace("_", " ").title(),
        "focus": focus,
        "personality": personality,
        "pitch": pitch,
    }
    
    offer_chain = offer_prompt | dragon_llm
    response = offer_chain.invoke(input_data)

    # Save current offer into state
    state["current_offer"] = response.content

    return {"messages": [response]}


def entrepreneur_counter_node(state: State):
    dragon_offer = state["messages"][-1].content

    # Entrepreneur counter logic
    entrepreneur_chain = entrepreneur_counter_offer_prompt | entrepreneur_llm
    response = entrepreneur_chain.invoke({"offer": dragon_offer})

    # Save counter offer into state
    state["entrepreneur_counter_offer"] = response.content

    return {"messages": [response]}


def dragon_negotiation_node(state: State):
    counter_offer = state["messages"][-1].content

    dragon_name = state["current_dragon"]
    dragon_info = dragons[dragon_name]
    dragon_llm = dragon_info["llm"]
    negotiation_prompt_obj = dragon_negotiation_prompt

    focus = dragon_info.get("focus", "general business")
    personality = dragon_info.get("personality", "professional")

    input_data = {
        "dragon_name": dragon_name.replace("_", " ").title(),
        "focus": focus,
        "personality": personality,
        "counter_proposal": counter_offer,
    }

    negotiation_chain = negotiation_prompt_obj | dragon_llm
    response = negotiation_chain.invoke(input_data)

    return {"messages": [response]}


# Define workflow
builder = StateGraph(State)

builder.add_node("dragon_evaluation", dragon_evaluation_node)
builder.add_node("entrepreneur_response", entrepreneur_response_node)
builder.add_node("dragon_offer", dragon_offer_node)
builder.add_node("entrepreneur_counter", entrepreneur_counter_node)
builder.add_node("dragon_negotiation", dragon_negotiation_node)

builder.add_edge(START, "dragon_evaluation")
builder.add_edge("dragon_evaluation", "entrepreneur_response")
builder.add_edge("entrepreneur_response", "dragon_offer")
builder.add_edge("dragon_offer", "entrepreneur_counter")
builder.add_edge("entrepreneur_counter", "dragon_negotiation")
builder.add_edge("dragon_negotiation", END)

graph = builder.compile()

graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

def generate_pitch(industry: str):
    entrepreneur_chain = entrepreneur_pitch_prompt | entrepreneur_llm
    response = entrepreneur_chain.invoke({"industry": industry})
    return response

template_industry = "fintech"
pitch = generate_pitch(template_industry)
print(pitch.content)

pitch_text = pitch.content
input_message = HumanMessage(content=pitch_text)

state = State(
    messages=[input_message],
    industry=template_industry,
    pitch=pitch_text,
    summary="",
)

def pick_best_dragon(state: State):
    pitch = state["pitch"].lower()

    scores = {}
    for dragon_name, info in dragons.items():
        focus = info["focus"].lower()
        
        if focus in pitch:
            scores[dragon_name] = 2
        elif any(word in pitch for word in focus.split()):
            scores[dragon_name] = 1
        else:
            scores[dragon_name] = 0

    best_dragon = max(scores, key=lambda name: scores[name])
    
    print(f"Selected {best_dragon} based on pitch focus match.")

    return best_dragon 


# Pick best dragon based on pitch
selected_dragon = pick_best_dragon(state)
state["current_dragon"] = selected_dragon

for chunk in graph.stream(state, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
#     chunk["messages"][-1].pretty_print()