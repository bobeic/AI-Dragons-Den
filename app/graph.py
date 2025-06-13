from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from app.state import State
from app.nodes import (
    # dragon_evaluation_node,
    user_response_to_evaluation,
    dragon_offer_node,
    user_response_to_offer,
    dragon_negotiation_node,
    make_dragon_evaluation_node,
    make_user_interrupt_node
)

from app.dragons.dragons import dragons

builder = StateGraph(State)
for dragon in dragons:
    builder.add_node(f"{dragon}_evaluation", make_dragon_evaluation_node(dragon))
    builder.add_node(f"{dragon}_user_response", make_user_interrupt_node(dragon))

# Now wire them together in the order from dragon_order
dragon_list = list(dragons.keys())

for i in range(len(dragon_list) - 1):
    builder.add_edge(f"{dragon_list[i]}_user_response", f"{dragon_list[i+1]}_evaluation")

# builder.add_node("dragon_evaluation", dragon_evaluation_node)
# builder.add_node("user_response_to_evaluation", user_response_to_evaluation)
builder.add_node("dragon_offer", dragon_offer_node)
builder.add_node("user_response_to_offer", user_response_to_offer)
builder.add_node("dragon_negotiation", dragon_negotiation_node)

# builder.add_edge(START, "dragon_evaluation")
builder.add_edge(START, f"{dragon_list[0]}_evaluation")
builder.add_edge(f"{dragon_list[-1]}_user_response", "dragon_offer")

# builder.add_edge("dragon_evaluation", "user_response_to_evaluation")
# builder.add_edge("user_response_to_evaluation", "dragon_offer")
# builder.add_edge("dragon_offer", "user_response_to_offer")
# builder.add_edge("user_response_to_offer", "dragon_negotiation")

graph = builder.compile(checkpointer=MemorySaver())
