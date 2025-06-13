import streamlit as st
import requests
import os

API_URL = "http://localhost:8000"  

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

st.title("🧠 Dragons' Den AI")
st.write("Pitch your idea and negotiate with AI dragons.")

# New pitch input
if len(st.session_state.messages) == 0:
    user_pitch = st.text_area("Write your pitch to the dragons:", height=200)
    if st.button("Pitch to the Dragons 🐉"):
        if user_pitch.strip() == "":
            st.warning("Please enter a pitch before submitting.")
        else:
            st.session_state.messages.append({"type": "human", "content": user_pitch})
            res = requests.post(f"{API_URL}/start_conversation", json={"pitch": user_pitch})
            if res.status_code == 200:
                data = res.json()
                st.session_state.messages += data["messages"]
                st.session_state.thread_id = data["thread_id"]
            else:
                st.error("Something went wrong starting your pitch.")

if len(st.session_state.messages) >= 2:
    user_reply = st.text_area("Your reply to the dragons:", key="reply")

    if st.button("Send Reply"):
        res = requests.post(f"{API_URL}/user_reply", json={
            "message": user_reply,
            "thread_id": st.session_state.thread_id
        })
        if res.status_code == 200:
            data = res.json()
            st.session_state.messages += data["messages"]
        else:
            st.error("Error sending your reply. Try again.")


# Show message history
if st.session_state.messages:
    st.markdown("### Conversation")
    for msg in st.session_state.messages:
        type = msg["type"]
        content = msg["content"]

        if type == "human":
            st.chat_message("You").write(content)
        else:
            # Dragon response
            with st.chat_message("Dragon"):
                st.write(content)

                # 🔊 Play audio if available
                if "audio_path" in msg:
                    try:
                        with open(msg["audio_path"], "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                    except Exception as e:
                        st.warning(f"Could not play audio: {e}")
