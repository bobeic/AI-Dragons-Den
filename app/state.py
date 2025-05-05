from langgraph.graph import MessagesState

class State(MessagesState):
    industry: str  # The industry of the pitch
    pitch: str     # The original business pitch
    summary: str = ""  # Optional: could be used to summarize conversation later
    current_dragon: str = ""  # Who is the active dragon
    current_offer: str = ""  # Dragon's offer (if any)
    entrepreneur_counter_offer: str = ""  # Entrepreneur's counter-offer (if any)
