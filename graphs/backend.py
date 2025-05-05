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
from langgraph.types import interrupt, Command
from typing import Literal
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


def dragon_evaluation_node(state: State) -> Command[Literal["user_response_to_evaluation"]]:
    pitch = state["messages"][-1].content
    print("evaluating")

    dragon_name = state["current_dragon"]
    dragon_info = dragons[dragon_name]
    dragon_llm = dragon_info["llm"]
    evaluation_prompt = dragon_initial_evaluation_prompt

    focus = dragon_info.get("focus", "general business")
    personality = dragon_info.get("personality", "professional")

    input_data = {
        "dragon_name": dragon_name.replace("_", " ").title(),
        "focus": focus,
        "personality": personality,
        "pitch": pitch,
    }

    evaluation_chain = evaluation_prompt | dragon_llm
    response = evaluation_chain.invoke(input_data)

    return Command(
        update={"messages": state["messages"] + [{"role": "assistant", "content": response.content}]},
        goto="user_response_to_evaluation"
        # goto="entrepreneur_response"
    )


def entrepreneur_response_node(state: State) -> Command[Literal["dragon_offer"]]:
    evaluation = state["messages"][-1].content

    entrepreneur_chain = entrepreneur_response_prompt | entrepreneur_llm
    response = entrepreneur_chain.invoke({"evaluation": evaluation})

    return Command(
        update={"messages": state["messages"] + [{"role": "user", "content": response.content}]},
        goto="dragon_offer"
    )

def dragon_offer_node(state: State) -> Command[Literal["user_response_to_offer"]]:
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

    return Command(
        update={
            "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
            "current_offer": response.content,
        },
        goto="user_response_to_offer"
    )


def entrepreneur_counter_node(state: State) -> Command[Literal["dragon_negotiation"]]:
    dragon_offer = state["messages"][-1].content

    entrepreneur_chain = entrepreneur_counter_offer_prompt | entrepreneur_llm
    response = entrepreneur_chain.invoke({"offer": dragon_offer})

    return Command(
        update={
            "messages": state["messages"] + [{"role": "user", "content": response.content}],
            "entrepreneur_counter_offer": response.content,
        },
        goto="dragon_negotiation"
    )


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

    return Command(
        update={"messages": state["messages"] + [{"role": "assistant", "content": response.content}]}
        # goto="END"
    )


def user_response_to_evaluation(state) -> Command[Literal["dragon_offer"]]:
    """Pause and collect human entrepreneur's real input."""
    user_input = interrupt(value="Waiting for entrepreneur's response...")
    print("interrupting...")

    # Update conversation state with user's message
    return Command(
        update={
            "messages": state["messages"] + [{"role": "user", "content": user_input}]
        },
        goto="dragon_offer"
    )


def user_response_to_offer(state) -> Command[Literal["dragon_negotiation"]]:
    """Pause and collect human entrepreneur's real input."""
    user_input = interrupt(value="Waiting for entrepreneur's response...")
    print("interrupting...")

    # Update conversation state with user's message
    return Command(
        update={
            "messages": state["messages"] + [{"role": "user", "content": user_input}]
        },
        goto="dragon_negotiation"
    )


# Define workflow
builder = StateGraph(State)

builder.add_node("dragon_evaluation", dragon_evaluation_node)
builder.add_node("user_response_to_evaluation", user_response_to_evaluation) 
# builder.add_node("entrepreneur_response", entrepreneur_response_node) 
builder.add_node("dragon_offer", dragon_offer_node)
# builder.add_node("entrepreneur_counter", entrepreneur_counter_node)
builder.add_node("user_response_to_offer", user_response_to_offer)
builder.add_node("dragon_negotiation", dragon_negotiation_node)
# builder.add_node(END)

builder.add_edge(START, "dragon_evaluation")
# builder.add_edge("dragon_evaluation", "user_response_to_evaluation") 
# # builder.add_edge("user_response_to_evaluation", "dragon_offer")  
# builder.add_edge("dragon_offer", "entrepreneur_counter")
# builder.add_edge("entrepreneur_counter", "dragon_negotiation")
# builder.add_edge("dragon_negotiation", END)


# builder = StateGraph(State)

# builder.add_node("dragon_evaluation", dragon_evaluation_node)
# builder.add_node("pause_for_response", user_response_to_evaluation) 
# # builder.add_node("entrepreneur_response", entrepreneur_response_node)
# builder.add_node("dragon_offer", dragon_offer_node)
# builder.add_node("entrepreneur_counter", entrepreneur_counter_node)
# builder.add_node("dragon_negotiation", dragon_negotiation_node)

# builder.add_edge(START, "dragon_evaluation")
# # builder.add_edge("dragon_evaluation", "entrepreneur_response")
# # builder.add_edge("entrepreneur_response", "dragon_offer")
# # builder.add_edge("dragon_evaluation", "pause_for_response")
# # builder.add_edge("pause_for_response", "dragon_offer")
# builder.add_edge("dragon_evaluation", "dragon_offer")
# builder.add_edge("dragon_offer", "entrepreneur_counter")
# builder.add_edge("entrepreneur_counter", "dragon_negotiation")
# builder.add_edge("dragon_negotiation", END)


from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
# graph = builder.compile(interrupt_before=["dragon_offer"], checkpointer=memory)
graph = builder.compile(checkpointer=memory)

# graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

def generate_pitch(industry: str):
    entrepreneur_chain = entrepreneur_pitch_prompt | entrepreneur_llm
    response = entrepreneur_chain.invoke({"industry": industry})
    return response

template_industry = "fintech"
pitch = generate_pitch(template_industry)

pitch_text = pitch.content
# pitch_text = input("Enter your pitch: ")
# pitch_text = "Good evening, Dragons. I'm here to tackle the major problem of financial inclusion in the fintech industry. Many underserved communities lack access to basic financial services, such as savings accounts and credit. My innovative solution is 'FinClude', a mobile app that uses AI-powered identity verification to onboard users quickly and securely. Our USP is our proprietary algorithm, which reduces the risk of fraud and increases approval rates by 30%. We'll generate revenue through a subscription-based model, offering users a range of financial services, including loans, insurance, and investment products. I'm seeking a $500,000 investment in exchange for 10% equity. Who's ready to join the financial revolution?"
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

import uuid
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}



for chunk in graph.stream(state, thread, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# Ask user for feedback on the evaluation
evaluation_response = input("Enter your reply to the dragons: ")
# evaluation_response = "Thank you for the thoughtful feedback. Our algorithm leverages alternative data sources and continual model retraining to stay accurate and fraud-resistant. We've already onboarded 10,000 users in East Africa and seen 40% month-on-month growth. On pricing, we're piloting a $3/month tier that includes micro-loans and savings tools. We've also begun early talks with regulators to ensure compliance from day one. I’m confident we’re positioned to scale and would love to address these points further."

for chunk in graph.stream(Command(
        resume=evaluation_response), thread, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# Ask user for feedback on the offer
offer_response = input("Enter your reply to the dragons: ")
# offer_response = "Thank you for the offer — I really appreciate your interest. We're targeting over 50 million underserved users across emerging markets, with acquisition costs currently at £2 per user and strong early traction. Projected revenue growth is 5x year-over-year, driven by bundled service uptake. I’m encouraged by your partnership ideas and open to negotiating terms further — would you consider coming up to £400,000 for 8% equity?"

for chunk in graph.stream(Command(
        resume=offer_response), thread, stream_mode="values"):
    chunk["messages"][-1].pretty_print()



# for event in graph.stream(state, stream_mode="values"):
#     event['messages'][-1].pretty_print()

# user_response = input("Enter your reply to the dragons: ")
# state["messages"].append(HumanMessage(content=user_response))

# # select new dragon based on user input can be same dragon as before for now

# for event in graph.stream(None, stream_mode="values"):
#     event['messages'][-1].pretty_print()


