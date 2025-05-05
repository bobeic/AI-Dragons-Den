from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from app.state import State
from app.nodes import (
    dragon_evaluation_node,
    user_response_to_evaluation,
    dragon_offer_node,
    user_response_to_offer,
    dragon_negotiation_node
)

builder = StateGraph(State)
builder.add_node("dragon_evaluation", dragon_evaluation_node)
builder.add_node("user_response_to_evaluation", user_response_to_evaluation)
builder.add_node("dragon_offer", dragon_offer_node)
builder.add_node("user_response_to_offer", user_response_to_offer)
builder.add_node("dragon_negotiation", dragon_negotiation_node)

builder.add_edge(START, "dragon_evaluation")
# builder.add_edge("dragon_evaluation", "user_response_to_evaluation")
# builder.add_edge("user_response_to_evaluation", "dragon_offer")
# builder.add_edge("dragon_offer", "user_response_to_offer")
# builder.add_edge("user_response_to_offer", "dragon_negotiation")

graph = builder.compile(checkpointer=MemorySaver())
