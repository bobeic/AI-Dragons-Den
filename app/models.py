from langchain.chat_models import init_chat_model

def get_chat_model(model_name: str, provider: str = "groq", temperature: float = 0.0, streaming: bool = True):
    """
    Utility to initialize a chat model with common defaults.
    """
    return init_chat_model(
        model_name,
        model_provider=provider,
        temperature=temperature,
        streaming=streaming,
        max_retries=2,  # optional, helps for reliability
    )
