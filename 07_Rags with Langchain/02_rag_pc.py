
import streamlit as st
from langchain_community.vectorstores import Pinecone
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.schema import Document
import pinecone
from pinecone import Pinecone

# Initialize Streamlit app
st.title("RAG System Powered by Streamlit")
st.write("Ask a question and get relevant answers from the knowledge base.")

# Initialize Pinecone and embeddings
PINECONE_API_KEY = ""
PINECONE_ENVIRONMENT = "us-east-1"
GOOGLE_GEMINI_API_KEY = ""

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = "gemini-rag-index"

# Check if the index already exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine"
    )

index = pc.Index(index_name)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=GOOGLE_GEMINI_API_KEY
)

# Input field for user question
question = st.text_input("Enter your question:")

if st.button("Ask"):  
    try:
        # Generate embeddings for the question
        query_vector = embeddings.embed_query(question)
        
        # Retrieve relevant documents from Pinecone
        results = index.query(query_vector, top_k=5, include_metadata=True)
        
        # Construct response from retrieved documents
        retrieved_docs = [result["metadata"]["text"] for result in results["matches"]]
        
        # Display the answer
        answer = "\n".join(retrieved_docs) if retrieved_docs else "No relevant information found."
        st.write("**Answer:**", answer)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Developed by **Sheema Masood** and powered by **Streamlit**")
