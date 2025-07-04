import streamlit as st
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import tempfile
import os

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# UI Title
st.title("📄 Resume Analyzer Bot (RAB)")
st.markdown("Upload a PDF, ask questions, and get LLM-powered answers!")

# Input: OpenAI API key
api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# Upload PDF
uploaded_file = st.file_uploader("📁 Upload your PDF file", type=["pdf"])

# Function to chunk data
def get_data_chunks(text, chunk_size):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Create embeddings using OpenAI API
def get_embeddings(text):
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Process PDF and create ChromaDB
if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Read PDF content
    knowledge_base = ""
    with fitz.open(tmp_file_path) as doc:
        for page in doc:
            knowledge_base += page.get_text()

    # Display extracted text (optional)
    with st.expander("📖 View Extracted Text"):
        st.text(knowledge_base)

    # Chunk the content
    st.info("📦 Chunking PDF content...")
    knowledge_chunks = get_data_chunks(knowledge_base, chunk_size=20)

    # Setup ChromaDB
    #chroma_client = chromadb.Client(Settings(persist_directory="./chroma_data_store"))
    chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",persist_directory=None))  # No disk writes

    
    
    collection_name = "knowledge_base_resume"

    # List existing collections
    existing_collections = [col.name for col in chroma_client.list_collections()]

    # Delete if exists
    if collection_name in existing_collections:
        chroma_client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'.")

    # Create fresh collection
    collection = chroma_client.create_collection(name=collection_name)
    print(f"Created new collection '{collection_name}'.")

    st.success("✅ PDF processed successfully!")

    

    # Add to vector DB
    st.info("🔍 Generating embeddings and storing in ChromaDB...")
    for i, chunk in enumerate(knowledge_chunks):
        collection.add(
            ids=[f"chnk-{i+1}"],
            documents=[chunk],
            embeddings=[get_embeddings(chunk)]
        )

    st.success("✅ Vector store created successfully!")

    # Accept user prompt
    user_prompt = st.text_input("💬 Ask a question about the PDF")

    if user_prompt:
        client = OpenAI(api_key=api_key)
        user_query_embedding = get_embeddings(user_prompt)

        # Query ChromaDB
        results = collection.query(query_embeddings=[user_query_embedding], n_results=2)
        top_chunks = ", ".join(results["documents"][0])

        # Send to GPT
        st.info("🧠 Generating AI response...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"You are a resume analyser. Given: {top_chunks} and the question: '{user_prompt}', provide a helpful answer with a human touch."
                }
            ]
        )

        st.subheader("📝 AI Response")
        st.write(response.choices[0].message.content)

        # Cleanup temp file
        os.remove(tmp_file_path)
