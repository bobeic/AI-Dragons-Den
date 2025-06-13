from langgraph.types import Command, interrupt
from typing import Literal

from app.state import State
from app.dragons.dragons import dragons
from app.models import get_chat_model

from app.prompts.entrepreneur_prompts import (
    entrepreneur_response_prompt,
    entrepreneur_counter_offer_prompt,
    entrepreneur_pitch_prompt
)

from app.prompts.dragon_prompts import (
    dragon_initial_evaluation_prompt,
    dragon_offer_making_prompt,
    dragon_negotiation_prompt
)

entrepreneur_llm = get_chat_model(model_name="llama3-8b-8192", provider="groq")


def dragon_evaluation_node(state: State) -> Command[Literal["user_response_to_evaluation"]]:
    pitch = state["messages"][-1].content
    print("dragon evaluation_node")

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
    print("dragon offer_node")
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
    print("dragon negotiation_node")
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
    )


def user_response_to_evaluation(state) -> Command[Literal["dragon_offer"]]:
    """Pause and collect human entrepreneur's real input."""
    print("user_response_to_evaluation")
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
    print("user_response_to_offer")
    user_input = interrupt(value="Waiting for entrepreneur's response...")
    print("interrupting...")

    # Update conversation state with user's message
    return Command(
        update={
            "messages": state["messages"] + [{"role": "user", "content": user_input}]
        },
        goto="dragon_negotiation"
    )

def make_user_interrupt_node(dragon_name: str):
    def node(state: State) -> Command:
        user_input = interrupt(value=f"Waiting for reply to {dragon_name}...")
        return Command(
            update={"messages": state["messages"] + [{"role": "user", "content": user_input}]}
        )
    return node


def make_dragon_evaluation_node(dragon_name: str):
    def node(state: State) -> Command:
        dragon_info = dragons[dragon_name]
        dragon_llm = dragon_info["llm"]

        input_data = {
            "dragon_name": dragon_name.replace("_", " ").title(),
            "focus": dragon_info.get("focus", "general business"),
            "personality": dragon_info.get("personality", "professional"),
            "pitch": state["pitch"],
        }

        evaluation_chain = dragon_initial_evaluation_prompt | dragon_llm
        response = evaluation_chain.invoke(input_data)

        return Command(
            update={
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "name": dragon_name,
                    "content": response.content
                }],
                "current_dragon": dragon_name,
            },
            goto=f"{dragon_name}_user_response"
        )
    return node

def dragon_router_node(state: State):
    if state["current_index"] >= len(state["dragon_order"]):
        return Command(goto="dragon_offer")  # or next phase

    next_dragon = state["dragon_order"][state["current_index"]]
    return Command(
        update={"current_dragon": next_dragon},
        goto="dragon_response"
    )


def dragon_response_node(state: State):
    dragon = dragons[state["current_dragon"]]
    prompt = dragon_initial_evaluation_prompt | dragon["llm"]
    input_data = {
        "dragon_name": state["current_dragon"],
        "pitch": state["pitch"],
        "focus": dragon.get("focus", "general"),
        "personality": dragon.get("personality", "default")
    }
    result = prompt.invoke(input_data)
    return Command(
        update={
            "messages": state["messages"] + [{
                "role": "assistant",
                "name": state["current_dragon"],
                "content": result.content
            }]
        },
        goto="user_response_to_dragon"
    )

def user_response_to_dragon(state: State):
    user_input = interrupt(value="Waiting for entrepreneur...")
    return Command(
        update={
            "messages": state["messages"] + [{"role": "user", "content": user_input}],
            "current_index": state["current_index"] + 1
        },
        goto="dragon_router"
    )
