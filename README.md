#ChatBot
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
3. Run `streamlit run app.py`
