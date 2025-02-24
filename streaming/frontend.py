import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000"

st.title("AI Dragon's Den")

# Input industry from user
industry = st.text_input("Enter an industry:", "fitness technology")

# Ensure session state exists for pitch and response
if "pitch" not in st.session_state:
    st.session_state["pitch"] = ""

if "dragon_response" not in st.session_state:
    st.session_state["dragon_response"] = ""

# Button to generate pitch
if st.button("Generate Pitch"):
    st.subheader("Entrepreneur's Pitch:")

    # Stream response from FastAPI
    pitch_placeholder = st.empty()
    pitch_response = ""

    response = requests.post(
        f"{BACKEND_URL}/generate_pitch", json={"industry": industry}, stream=True
    )

    for chunk in response.iter_content(chunk_size=1024):
        text = chunk.decode("utf-8")
        pitch_response += text
        pitch_placeholder.markdown(pitch_response)  # Update UI in real-time

    st.session_state["pitch"] = pitch_response  # Store for evaluation
    st.session_state["dragon_response"] = ""  # Clear previous investor response

# Display the saved pitch even after evaluation
# if st.session_state["pitch"]:


# Button to evaluate pitch
if st.session_state["pitch"] and st.button("Evaluate Pitch"):
    st.subheader("Entrepreneur's Pitch:")
    st.markdown(st.session_state["pitch"])
    st.subheader("Investor's Response:")

    # Stream response from FastAPI
    response_placeholder = st.empty()
    dragon_response = ""

    response = requests.post(
        f"{BACKEND_URL}/evaluate_pitch",
        json={"idea": st.session_state["pitch"]},
        stream=True,
    )

    for chunk in response.iter_content(chunk_size=1024):
        text = chunk.decode("utf-8")
        dragon_response += text
        response_placeholder.markdown(dragon_response)  # Update UI in real-time

    st.session_state["dragon_response"] = dragon_response  # Store response

# Display the investor's response even after page updates
# if st.session_state["dragon_response"]:
#     st.subheader("Investor's Response:")
#     st.markdown(st.session_state["dragon_response"])
