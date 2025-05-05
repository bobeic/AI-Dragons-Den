import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Change this if deploying

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
                # print("data:")
                # print(data)
                # print("data['thread_id']")
                # print(data['thread_id'])
                st.session_state.messages += data["messages"]
                st.session_state.thread_id = data["thread_id"]
            else:
                st.error("Something went wrong starting your pitch.")

if len(st.session_state.messages) >= 2:
    user_reply = st.text_area("Your reply to the dragons:", key="reply")
    print(st.session_state.thread_id)

    if st.button("Send Reply"):
        res = requests.post(f"{API_URL}/user_reply", json={
            "message": user_reply,
            "thread_id": st.session_state.thread_id
        })
        if res.status_code == 200:
            data = res.json()
            st.session_state.messages += data["messages"]
            # st.experimental_rerun()
        else:
            st.error("Error sending your reply. Try again.")


# Show message history
if st.session_state.messages:
    st.markdown("### Conversation")
    # print("messages:")
    # print(st.session_state.messages)
    for msg in st.session_state.messages:
        # print("msg:")
        # print(msg)
        type = msg["type"]
        content = msg["content"]
        if type == "human":
            st.chat_message("You").write(content)
        else:
            st.chat_message("Dragon").write(content)

# if len(st.session_state.messages) >= 2 and "Waiting for entrepreneur's response..." in st.session_state.messages[-1]["content"]:
#     user_reply = st.text_area("Your reply to the dragons:", key="reply")
#     if st.button("Send Reply"):
#         res = requests.post(f"{API_URL}/user_reply", json={
#             "message": user_reply
#         })
#         if res.status_code == 200:
#             data = res.json()
#             st.session_state.messages += data["messages"]
#             st.experimental_rerun()
#         else:
#             st.error("Error sending your reply. Try again.")
