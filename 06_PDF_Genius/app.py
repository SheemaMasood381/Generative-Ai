import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="chroma_db")
    vector_store.persist()

def get_gemini_response(prompt):
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    response = chat_model.predict(prompt)
    return response

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    docs = new_db.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {user_question}\nAnswer:"
    response = get_gemini_response(prompt)
    st.write("Reply: ", response)

def summarize_text(text, length="medium"):
    summary_prompt = f"Summarize the following text in a {length} manner:\n\n" + text
    return get_gemini_response(summary_prompt)

def main():
    st.set_page_config("PDF Genius - Chat & Summarizer")
    st.title("📚 PDF Genius")
    st.markdown("### Chat with your PDFs and generate summaries with Gemini! 💡") 

    with st.sidebar:
        st.title("📌 Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state["raw_text"] = raw_text
                st.success("Processing Complete ✅")

    tab1, tab2 = st.tabs(["💬 Chat with PDF", "📄 Document Summarizer"])

    with tab1:
        st.header("💬 Chat with PDF")
        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question:
            user_input(user_question)

    with tab2:
        st.header("📄 Document Summarizer")
    
        if "raw_text" in st.session_state:
            length = st.selectbox("Select Summary Length", ["short", "medium", "detailed"], index=1)
        
            if st.button("Summarize Document"):
                summary = summarize_text(st.session_state["raw_text"], length)
                st.subheader("Summary:")
                st.write(summary)
        else:
            st.warning("⚠ Please upload and process a PDF first from the sidebar.")

        # Footer
    st.markdown("---")
    st.markdown("Developed by **Sheema Masood** | Powered by **Streamlit**")
    st.markdown("🚀 Running at [Hugging Face Spaces](https://huggingface.co/spaces/SheemaMasood/PDFGenius)")


if __name__ == "__main__":
    main()
