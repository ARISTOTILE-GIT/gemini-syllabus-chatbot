# --- STEP 4: Namma Main Chatbot Code (v19 - The FINAL HYBRID Table Fix!) ---
import streamlit as st
# --- MAATHAM 1 (CHANGE): Namma ippo 'PyPDFLoader' ah import panrom ---
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os, traceback, tempfile # 'tempfile' mukkiyam

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="Syllabus Chatbot (Gemini-Hybrid)", page_icon="🤖")

# --- Title and Description ---
st.title("📚 IT Syllabus Chatbot (Gemini-Hybrid)")
st.write("Machi, idhu namma 'Hybrid' RAG Chatbot. Gemini (FREE) + HuggingFace (FREE) + Streamlit (FREE)!")
st.write("PDF ah upload pannitu, ethu venalum kelu!")

# --- API Key Check (Load from Streamlit Secrets) ---
# Idhu thaan namma Colab code kkum, deploy code kkum periya vithyasam!
# Namma key ah 'st.secrets' la irunthu edukurom.
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("ERROR: GOOGLE_API_KEY secret ah set pannala! GitHub la irunthu Streamlit settings ku po.")
    st.stop() # App ah ingaye stop panniru

# API key ah set pannu
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- RAG Chain Function (Idha cache panniralam) ---
# '@st.cache_resource' podurathaala, PDF ah ovvoru thadavayum process panna venam. Oru thadava pannatha veche use pannikum.
@st.cache_resource(show_spinner="PDF ah padikiren... Vector store create panren...")
def get_rag_chain(_uploaded_file):
    try:
        # 1. Upload panna file ah 'temp' ah save panrom (Appo thaan PyPDFLoader atha padikkum)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(_uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # 2. PyPDFLoader moolama PDF ah load panrom (Table Fix!)
        loader = PyPDFLoader(pdf_path)
        
        # 3. PDF ah split panrom
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
        
        # 8. Prompt (kelvi kekura template)
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. You MUST answer the question strictly using the following context. If the answer is not in the context, state that you cannot find the specific information.\n\nContext:\n{context}\n\nQuestion: {input}\n\nAnswer:"
        )
        
        # 9. 'Wiring' (RAG Chain)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        print("✅ RAG Chain created successfully!")
        return retrieval_chain

    except Exception as e:
        st.error(f"Error appo PDF process panrom: {e}")
        return None
    finally:
        # Temp file ah delete panniru
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)

# --- Streamlit UI Logic ---

# 1. PDF Upload UI (Sidebar la)
with st.sidebar:
    st.header("1. PDF ah Upload Pannu")
    uploaded_file = st.file_uploader("Unnoda 'syllabus.pdf' file ah inga podu", type="pdf")

# 2. Chat History Setup (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Pazhaya message ah display pannu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Main workflow
if uploaded_file:
    # PDF process panni, RAG chain ah vaangurom
    rag_chain = get_rag_chain(uploaded_file)
    
    if rag_chain:
        # Ippo, user kitta kelvi kekurathukku chat box ah kaatrom
        if user_question := st.chat_input("Unnoda kelvi enna?"):
            
            # User message ah save panni display pannu
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Bot oda badhil ah vaangu
            with st.chat_message("assistant"):
                with st.spinner("Pathil theduren... (Gemini is thinking...)"):
                    try:
                        # RAG chain ah run pannu
                        response = rag_chain.invoke({"input": user_question})
                        answer = response.get("answer", "⚠️ No answer found. Try rephasing.")
                        
                        # Bot message ah save panni display pannu
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error appo badhil theduren: {e}")
                        traceback.print_exc()

else:
    # PDF upload pannala na, message kaatu
    st.info("Please upload a PDF file in the sidebar to start the chat.")

