import streamlit as st
import requests

# Set the backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Streamlit UI
st.title("AI Dragon’s Den")

st.subheader("Entrepreneur Pitch Generator")

# Input field for industry
industry = st.text_input(
    "Enter an industry (e.g., fitness technology, fintech, AI, etc.)"
)

# Generates all in one go
if st.button("Generate Pitch"):
    if industry:
        with st.spinner("Generating pitch..."):
            response = requests.post(
                f"{BACKEND_URL}/generate_pitch", json={"industry": industry}
            )
            if response.status_code == 200:
                pitch_data = response.json()
                # print(data)
                pitch = pitch_data["idea"]["choices"][0]["message"]["content"]
                st.success("Entrepreneur's Pitch:")
                st.write(pitch)
            else:
                st.error("Failed to generate pitch. Please try again.")

        # Send pitch to be evaluated
        with st.spinner("Evaluating pitch..."):
            eval_response = requests.post(
                f"{BACKEND_URL}/evaluate_pitch", json={"idea": pitch}
            )
            if eval_response.status_code == 200:
                eval_data = eval_response.json()
                evaluation = eval_data["evaluation"]["choices"][0]["message"]["content"]
                st.success("Dragon's Response:")
                st.write(evaluation)
            else:
                st.error("Failed to evaluate pitch. Please try again.")
    else:
        st.warning("Please enter an industry.")

