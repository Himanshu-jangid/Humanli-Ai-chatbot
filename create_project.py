import os
import zipfile

# Define the project structure and contents
project_files = {
    "requirements.txt": """streamlit
langchain
langchain-community
langchain-openai
langchain-text-splitters
faiss-cpu
beautifulsoup4
python-dotenv""",

    "app.py": """import streamlit as st
from utils.rag_logic import get_vector_store, get_rag_chain
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Humanli.ai Chatbot", layout="wide")
st.title("🌐 Website-Based Chatbot")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter Website URL", placeholder="https://example.com")
    if st.button("Index Website"):
        if website_url:
            with st.spinner("Crawling and Indexing..."):
                st.session_state.vector_store = get_vector_store(website_url)
                st.success("Indexing Complete!")
        else:
            st.error("Please enter a valid URL.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("Ask a question about the website:"):
    if "vector_store" not in st.session_state:
        st.info("Please index a website URL first.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            rag_chain = get_rag_chain(st.session_state.vector_store)
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.markdown(answer)
            
        from langchain_core.messages import HumanMessage, AIMessage
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=answer)
        ])""",

    "utils/rag_logic.py": """import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_vector_store(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_rag_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vector_store.as_retriever()

    contextualize_q_system_prompt = "Given a chat history and the latest user question, formulate a standalone question."
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. "
        "If the answer is not available on the provided website, respond exactly with: "
        "'The answer is not available on the provided website.' "
        "Do not use external knowledge or hallucinate."
        "\\n\\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)""",

    "README.md": """# Humanli.ai AI/ML Engineer Assignment
## Overview
An AI-powered chatbot that crawls a website and answers questions strictly based on its content.

## Architecture
- **Framework**: LangChain & Streamlit
- **LLM**: GPT-4o-mini (chosen for precision in grounded tasks)
- **Vector DB**: FAISS (for fast similarity search and persistence)
- **Embedding**: OpenAI Embeddings

## Setup
1. `pip install -r requirements.txt`
2. Create a `.env` file with `OPENAI_API_KEY=your_key_here`
3. Run `streamlit run app.py`""",
    
    ".env": "OPENAI_API_KEY=your_api_key_here"
}

def create_zip():
    zip_name = "humanli_assignment.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file_path, content in project_files.items():
            # Create subdirectories if they don't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True) if "/" in file_path else None
            # Write file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            # Add to zip
            zipf.write(file_path)
    print(f"Successfully created {zip_name}")

if __name__ == "__main__":
    create_zip()