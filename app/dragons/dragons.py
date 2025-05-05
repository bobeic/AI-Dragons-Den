import os
from dotenv import load_dotenv

from app.models import get_chat_model
from app.dragons.dragon_bios import dragon_bios
from app.rag import create_dragon_vector_store
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_model = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

dragon_names = list(dragon_bios.keys())
dragon_llms = {name: get_chat_model("llama3-8b-8192", provider="groq") for name in dragon_names}
dragon_vectorstores = {
    name: create_dragon_vector_store(name, dragon_bios[name], embedding_model=embedding_model)
    for name in dragon_names
}

dragons = {
    name: {
        "llm": dragon_llms[name],
        "vectorstore": dragon_vectorstores[name],
        "focus": dragon_bios[name][0]["focus"],
        "personality": dragon_bios[name][0]["personality"],
    }
    for name in dragon_names
}

def pick_best_dragon(pitch: str) -> str:
    # pitch = state["pitch"].lower()

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