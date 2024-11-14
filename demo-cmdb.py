import pandas as pd
import streamlit as st
import cohere
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Step 1: Load CSV Data
data_file = "sample_cmdb_data.csv"
df = pd.read_csv(data_file)

# Step 3: Split each row of data directly for Better Precision
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_documents = []
for _, row in df.iterrows():
    content = (
        f"Hostname: {row['Hostname']}\n"
        f"IP Address: {row['IP_Address']}\n"
        f"Operating System: {row['Operating_System']}\n"
        f"CPU: {row['CPU']}\n"
        f"Memory (GB): {row['Memory_GB']}\n"
        f"Storage (GB): {row['Storage_GB']}\n"
        f"Location: {row['Location']}\n"
        f"Environment: {row['Environment']}\n"
        f"Status: {row['Status']}\n"
        f"Owner: {row['Owner']}\n"
        f"Purchase Date: {row['Purchase_Date']}\n"
        f"Warranty Expiry: {row['Warranty_Expiry']}\n"
        f"Last Patched Date: {row['Last_Patched_Date']}\n"
    )
    split_documents.extend(text_splitter.split_text(content))

# Step 4: Set Up Vector Database (FAISS) using Local Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(split_documents, embedding)

# Step 5: Use the Retriever for Document Matching
retriever = vectorstore.as_retriever()

# Function to query based on content using retriever and Cohere API for accuracy
def answer_question(question: str):
    docs = retriever.invoke(question, k=5)  # Retrieve top 5 relevant documents
    if docs:
        # Concatenate the content of the top 5 documents
        context = "\n".join([doc.page_content for doc in docs])
        print(context)
        
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
st.title("CMDB Data Question Answering System")
st.write("Ask a question about CMDB data (e.g., 'What is the IP address of Cassiopeia?')")

question = st.text_input("Enter your question:")
if question:
    answer = answer_question(question)
    st.write("Answer:", answer)
