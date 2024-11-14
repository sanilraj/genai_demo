import pandas as pd
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere
import os
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Step 1: Load CSV Data
data_file = "sample_cmdb_data.csv"
df = pd.read_csv(data_file)

# Step 2: Split each row of data directly for Better Precision
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

# Step 3: Set Up Vector Database (FAISS) using Local Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(split_documents, embedding)

# Step 4: Use the Retriever for Document Matching
retriever = vectorstore.as_retriever()

# Step 5: Set up Cohere LLM for Question Answering
llm = Cohere(cohere_api_key=os.getenv("COHERE_API_KEY"), model="command-xlarge-nightly")
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Function to query based on content using retriever and LangChain QA Chain
def answer_question(question: str):
    docs = retriever.get_relevant_documents(question)  # Retrieve top relevant documents
    if docs:
        try:
            # Use LangChain QA Chain to get an accurate answer
            answer = qa_chain.run(input_documents=docs, question=question)
            return answer.strip()
        except Exception as e:
            logging.error(f"LangChain Cohere QA error: {e}")
            # Fallback to using the retrieved documents directly
            context = "\n".join([doc.page_content for doc in docs])
            return f"LangChain QA error. Here are the top results:\n\n{context}"
    else:
        return "No relevant information found."

# Streamlit UI for asking questions
st.title("CMDB Data Question Answering System")
st.write("Ask a question about CMDB data (e.g., 'What is the IP address of Cassiopeia?')")

question = st.text_input("Enter your question:")
if question:
    answer = answer_question(question)
    st.write("Answer:", answer)
