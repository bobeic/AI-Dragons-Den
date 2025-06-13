import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.types import Command
from app.graph import graph
from app.state import State
from app.dragons.dragons import pick_best_dragon, rank_dragons_by_relevance
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from app.utils.tts import generate_tts  


app = FastAPI()

thread_store = {}

class PitchRequest(BaseModel):
    pitch: str

class UserReplyRequest(BaseModel):
    thread_id: str
    message: str

@app.post("/start_conversation")
def start_conversation(req: PitchRequest):
    input_message = HumanMessage(content=req.pitch)

    # state = State(
    #     messages=[input_message],
    #     industry="general",  # Optional – you can parse this from the pitch if needed
    #     pitch=req.pitch,
    #     summary="",
    # )

    state = State(
        messages=[input_message],
        pitch=req.pitch,
        summary="",
        dragon_order=rank_dragons_by_relevance(req.pitch),  # New
    )

    
    selected_dragon = pick_best_dragon(state)
    state["current_dragon"] = selected_dragon

    thread_id = str(uuid4())
    print(thread_id)

    thread = {"configurable": {"thread_id": thread_id}}
    print("thread_id:")
    print(thread_id)

    new_messages = []

    for step in graph.stream(state, thread, stream_mode="values"):
        print("step: ")
        msg = step["messages"][-1]
        if isinstance(msg, AIMessage):
            msg_type = "dragon"
            audio_path = asyncio.run(generate_tts(msg.content))  # Generate audio
            new_messages.append({
                    "type": msg_type,
                    "content": msg.content,
                    "audio_path": audio_path
            })
    return {
        "messages": new_messages,
        "thread_id": thread_id,
    }

@app.post("/user_reply")
def resume_conversation(req: UserReplyRequest):
    resume_cmd = Command(resume=req.message)

    thread = {"configurable": {"thread_id": req.thread_id}}
    print("req")
    print(req)
    print("req thread_id:")
    print(req.thread_id)

    new_messages = []

    for i, step in enumerate(graph.stream(resume_cmd, thread, stream_mode="values")):
        if i == 0:
            continue
        print("step:")
        print(step["messages"])
        msg = step["messages"][-1]
        if isinstance(msg, AIMessage):
            msg_type = "dragon"
        elif isinstance(msg, HumanMessage):
            msg_type = "human"
        new_messages.append({
                "type": msg_type,
                "content": msg.content
            })
    return {
        "messages": new_messages,
    }

@app.get("/messages/{thread_id}")
def get_conversation(thread_id: str):
    state = thread_store.get(thread_id)
    if not state:
        return {"error": "Thread not found"}
    return {"thread_id": thread_id, "messages": state["messages"]}
