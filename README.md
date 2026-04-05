# Chat-with-my-pdf-AI
This is a Streamlit-based RAG (Retrieval-Augmented Generation) application that allows users to upload multiple PDF documents and ask questions about their content. The app uses Google Gemini 1.5 Flash for natural language processing and FAISS for efficient vector similarity search.

## Features
Multi-PDF Support: Upload and process multiple documents simultaneously.

Smart Text Chunking: Uses RecursiveCharacterTextSplitter for context-aware document splitting.

Vector Search: High-performance similarity search using FAISS and Google Generative AI Embeddings.

AI Conversations: Context-aware answering using the Gemini-1.5-Flash model.

## Tech Stack
Frontend: Streamlit

AI Model: Google Gemini API

Orchestration: LangChain v1.2+

Vector Store: FAISS

Environment: Python 3.13 (2026 Standard)


## Prerequisites
A Google API Key (Get it from Google AI Studio)

Python 3.13+ installed on your system.

## How it Works
Upload: Use the sidebar to upload one or more PDF files.

Process: The app extracts text, splits it into chunks, and creates a searchable vector database.

Ask: Type your question in the main text box.

Answer: The AI retrieves relevant sections from your PDFs and generates a precise answer.

## Running the App
To ensure the app uses the correct Python environment (especially if you have Anaconda installed), run it using the full path:
python -m streamlit run app.py

## author
srilipta mishra
sneha shaw
shriya subudhi

