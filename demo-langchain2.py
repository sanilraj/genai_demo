import pandas as pd
import streamlit as st
import cohere
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Step 1: Load CSV Data
data_file = "people-101.csv"
df = pd.read_csv(data_file)

# Step 2: Convert Rows to Documents
documents = []
for _, row in df.iterrows():
    content = (
        f"User ID: {row['User Id']}\n"
        f"Name: {row['First Name']} {row['Last Name']}\n"
        f"Sex: {row['Sex']}\n"
        f"Email: {row['Email']}\n"
        f"Phone: {row['Phone']}\n"
        f"Date of Birth: {row['Date of birth']}\n"
        f"Job Title: {row['Job Title']}\n"
    )
    documents.append(Document(page_content=content))

# Step 3: Split Documents for Better Precision
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

# Step 4: Set Up Vector Database (FAISS) using Local Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_documents, embedding)

# Step 5: Use the Retriever for Document Matching
retriever = vectorstore.as_retriever()

# Function to query based on content using retriever and Cohere API for accuracy
def answer_question(question: str):
    docs = retriever.invoke(question, k=5)  # Retrieve top 5 relevant documents
    if docs:
        # Concatenate the content of the top 5 documents
        context = "\n".join([doc.page_content for doc in docs])
        
        # Use Cohere API to get an accurate answer
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
        try:
            response = co.generate(
                model="command-xlarge-nightly",
                prompt=f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
                max_tokens=150,
                temperature=0.5
            )
            return response.generations[0].text.strip()
        except Exception as e:
            logging.error(f"Cohere API error: {e}")
            # Fallback to using the retrieved documents directly
            return f"Cohere API error. Here are the top results:\n\n{context}"
    else:
        return "No relevant information found."

# Streamlit UI for asking questions
st.title("People Data Question Answering System")
st.write("Ask a question about people data (e.g., 'Who is the Games developer?')")

question = st.text_input("Enter your question:")
if question:
    answer = answer_question(question)
    st.write("Answer:", answer)
