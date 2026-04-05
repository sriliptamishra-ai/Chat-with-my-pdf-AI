import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# -------- LOAD API KEY --------
load_dotenv()
genai.configure(api_key=os.getenv("openai_api_key"))

# -------- CACHE EMBEDDINGS (FASTER) --------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------- PDF TEXT --------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# -------- TEXT CHUNKS --------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

# -------- VECTOR STORE --------
def get_vector_store(text_chunks):
    embeddings = load_embeddings()

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")

# -------- QA CHAIN --------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say:
    "answer is not available in the context"

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = prompt | model | StrOutputParser()
    return chain

# -------- USER INPUT --------
def user_input(user_question):
    embeddings = load_embeddings()

    try:
        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except:
        st.error(" Please upload and process PDF first")
        return

    docs = new_db.similarity_search(user_question)

    context = "\n".join([doc.page_content for doc in docs])

    chain = get_conversational_chain()

    response = chain.invoke({
        "context": context,
        "question": user_question
    })

    st.markdown("###  Reply")
    st.write(response)

# -------- MAIN --------
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")

    st.header("📄 Chat with Multiple PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title(" Menu")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success(" Processing Complete")
            else:
                st.warning("⚠️ Please upload at least one PDF")

# -------- RUN --------
if __name__ == "__main__":
    main()