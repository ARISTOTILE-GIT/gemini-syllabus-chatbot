# 🚀 Gemini-Hybrid RAG PDF Chatbot

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-Framework-orange)
![Gemini](https://img.shields.io/badge/Gemini-API-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

A **100% free, high-performance RAG (Retrieval-Augmented Generation)** chatbot that intelligently answers questions about any uploaded PDF document.

This project uses a cost-effective **hybrid architecture**, combining the power of the **free-tier Google Gemini API** for reasoning and the efficiency of **free Hugging Face models** for embeddings.  
Built entirely with **LangChain** and deployed live on **Streamlit Cloud**.

---

## 📸 Live Demo & App Preview

👉 **[Try it Live on Streamlit!](https://gemini-syllabus-chatbot-totz.streamlit.app/)** 👈  

---

## 🧠 Overview

This project was born from an *“Edison experiment”* — to build a **smart, reliable, and cost-free document chatbot**.

The main challenge: navigating the complex world of GenAI APIs — from billing (OpenAI/Google embeddings) and resource limits (Mistral 7B RAM crash on Colab) to dependency chaos (`langchain.chains` errors).

The **solution**: a **robust Hybrid RAG system** that strategically uses the best *free tools* for each stage of the process.

---

## ✨ Key Features

- 💬 **Chat with Any PDF:** Upload a syllabus, technical manual, or research paper — get instant, context-aware answers.  
- 🧭 **Intelligent & Honest:** The bot finds exact answers; if it’s not in the PDF, it’ll tell you.  
- 💸 **100% Free Architecture:**
  - Free **HuggingFace embeddings** for vectorization  
  - Free-tier **Gemini 2.5 Flash API** for response generation  
- 🧩 **Handles Complex PDFs:** Uses `PyPDFLoader` to extract text and tables efficiently (fixed in **v19!**)  
- 🌐 **Live & Deployed:** Fully containerized and hosted on Streamlit Cloud  

---

## ⚙️ How It Works: Hybrid RAG Architecture

Here’s what happens when you ask a question 👇

1. **UI & Upload (Streamlit):** You upload a PDF via the Streamlit interface.

2. **Load & Split (PyPDFLoader):** The PDF is securely loaded and split into small, indexed chunks.

3. **Embed (HuggingFace - FREE):** Each chunk is turned into a numerical vector using `all-MiniLM-L6-v2`.

4. **Store (FAISS):** These vectors are stored in-memory with **FAISS**, forming the “brain” of the PDF.

5. **Retrieve (LangChain):** LangChain retrieves the most relevant chunks based on your question.

6. **Generate (Gemini - FREE):** The system sends your question + relevant chunks to **Gemini 2.5 Flash**, which generates the final answer.

7. **Answer:** A smart, accurate response appears instantly in the chat UI.

---

## 💻 Tech Stack

| Component | Technology | 
 | ----- | ----- | 
| **Core Framework** | LangChain | 
| **LLM (Brain)** | Google Gemini 2.5 Flash (Free API Tier) | 
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | 
| **Vector Database** | FAISS-CPU | 
| **Web App & UI** | Streamlit | 
| **Deployment** | Streamlit Cloud | 
| **PDF Parsing** | PyPDFLoader, pypdfium2, pypdf | 
| **Dependencies** | transformers, sentence-transformers, langchain-core | 

## 🧰 How to Run Locally

### 1️⃣ Clone the Repository

git clone https://github.com/ARISTOTLE-GIT/gemini-syllabus-chatbot.git 
cd gemini-syllabus-chatbot

### 2️⃣ Create a Virtual Environment

python -m venv venv source venv/bin/activate 
On Windows: venv\Scripts\activate

### 3️⃣ Install Dependencies

pip install -r requirements.txt

### 4️⃣ Add Your API Key

Create a folder named `.streamlit` and inside it a file called `secrets.toml`:
GOOGLE_API_KEY = "AIzaSy...Your...Key...Here"

### 5️⃣ Run the App

streamlit run app.py


## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.


## 👨‍💻 Developer Info

**Developed by:** [Aristotile S](https://github.com/ARISTOTLE-GIT)

🎓 Passionate about AI, Generative Models, and Open-Source Innovation.

💬 *“Building smarter systems that help people learn, create, and innovate — one project at a time.”*
