# Main Chatbot Code 
import streamlit as st
# --- We are now importing 'PyPDFLoader' ---
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os, traceback, tempfile # 'tempfile' is important

st.set_page_config(
    page_title="Syllabus Chatbot (Gemini-Hybrid)", 
    page_icon="🤖",
    theme="light" 
)

# --- Title and Description (All English) ---
st.title("📚 IT Syllabus Chatbot (Gemini-Hybrid)")
st.write("This is our 'Hybrid' RAG Chatbot. Gemini (FREE) + HuggingFace (FREE) + Streamlit (FREE)!")
st.write("Upload a PDF and ask anything about it!")

# --- API Key Check (Load from Streamlit Secrets) ---
# This is the main difference between the Colab code and the deployment code!
# We are getting the key from 'st.secrets'.
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("ERROR: GOOGLE_API_KEY secret is not set! Go to Streamlit settings.")
    st.stop() # Stop the app here

# Set the API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- RAG Chain Function (Cached) ---
# '@st.cache_resource' ensures we don't re-process the PDF every time. 
# It uses the cached result.
@st.cache_resource(show_spinner="Reading PDF... Creating vector store...")
def get_rag_chain(_uploaded_file):
    try:
        # 1. Save the uploaded file as a 'temp' file (So PyPDFLoader can read it)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(_uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # 2. Load the PDF using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        
        # 3. Split the PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = loader.load_and_split(text_splitter)
        
        # 4. HuggingFace Embeddings (FREE)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
        
        # 5. Vector Store (FAISS)
        db = FAISS.from_documents(chunks, embedding=embeddings) 
        
        # 6. Retriever
        retriever = db.as_retriever()
        
        # 7. Gemini Chat 'Brain' (FREE TIER)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025", 
            temperature=0.3
        )
        
        # 8. Prompt (The template for asking questions)
        
        new_prompt_template = """
        You are a helpful assistant.

        First, try to answer the question using the 'Context' provided below.
        If the answer is not found in the context, it's okay, just use your general knowledge to answer the question.

        Context:
        {context}

        Question: {input}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(new_prompt_template)
        
        # 9. 'Wiring' (The RAG Chain)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        print("✅ RAG Chain created successfully!")
        return retrieval_chain

    except Exception as e:
        st.error(f"Error while processing PDF: {e}")
        return None
    finally:
        # Delete the temp file
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)

# --- Streamlit UI Logic ---

# 1. PDF Upload UI (In the Sidebar)
with st.sidebar:
    st.header("1. Upload your PDF")
    uploaded_file = st.file_uploader("Upload your 'syllabus.pdf' file here", type="pdf")

# 2. Chat History Setup (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Main workflow
if uploaded_file:
    # Process the PDF and get the RAG chain
    rag_chain = get_rag_chain(uploaded_file)
    
    if rag_chain:
        # Now, show the chat box to get the user's question
        if user_question := st.chat_input("What is your question?"):
            
            # Save and display the user message
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Get the bot's response
            with st.chat_message("assistant"):
                with st.spinner("Searching for an answer... (Gemini is thinking...)"):
                    try:
                        # Run the RAG chain
                        response = rag_chain.invoke({"input": user_question})
                        answer = response.get("answer", "⚠️ No answer found. Try rephrasing.")
                        
                        # Save and display the bot message
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error while fetching the answer: {e}")
                        traceback.print_exc()

else:
    # If no PDF is uploaded, show a message
    st.info("Please upload a PDF file in the sidebar to start the chat.")
