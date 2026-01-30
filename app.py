import streamlit as st
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
        ])