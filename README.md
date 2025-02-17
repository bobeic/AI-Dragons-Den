# Dragon's Den AI

## 📌 Project Overview
Dragon's Den AI is a simulation of the popular TV show "Dragon's Den," where entrepreneurs pitch startup ideas to seasoned investors. This project uses **LLMs (Large Language Models)** to generate realistic interactions between entrepreneurs and judges.

## 🚀 Features
- **Entrepreneur AI** generates startup pitches based on an industry.
- **Investor AI** evaluates pitches, providing strengths, weaknesses, and suggestions.
- **Streaming Support** for real-time interactions.
- **Hugging Face API Integration** with models like **Mistral 7B**.

## 🛠️ Technologies Used
- **Python**
- **LangChain (langchain_core, langchain_community)**
- **Hugging Face Inference API**

## ⚙️ Setup Instructions
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/bobeic/AI-Dragons-Den.git
cd dragons-den-ai
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys
- Get a Hugging Face API key from [Hugging Face](https://huggingface.co/settings/tokens)
- Create a `.env` file and add:
  ```
  HUGGINGFACEHUB_API_TOKEN=your_api_key_here
  ```

### 4️⃣ Run the Project
To run the project without streaming the text do the following:
- Setup up the backend:
```bash
cd no_streaming
uvicorn backend:app --reload
```
- In a different terminal, run the frontend:
```bash
cd no_streaming
streamlit run frontend.py
```

## 🎯 Example Interaction
### **Entrepreneur Pitch**
```
Industry: Fitness Technology
Entrepreneur: "We have developed a wearable that tracks hydration levels in real time."
```

### **Investor Response**
```
Dragon: "Interesting idea! What differentiates this from existing fitness trackers?"
```
