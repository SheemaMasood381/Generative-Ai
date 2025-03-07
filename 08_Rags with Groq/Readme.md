# RAG Document Q&A With Groq and Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** system using **Groq's Llama3** model and **Ollama embeddings**. The application is built with **Streamlit** and allows users to query research papers using an AI-powered Q&A system.

## Features
- Load and process PDF research papers.
- Generate vector embeddings using **Ollama's `mxbai-embed-large`** model.
- Store document vectors using **FAISS** for fast retrieval.
- Use **Groq's Llama3-8b-8192** model for intelligent responses.
- Provide contextual answers based on retrieved document segments.
- Display similar document sections for better reference.

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-groq-ollama.git
cd rag-groq-ollama
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project directory and add your API keys:
```
Groq_Api_Key=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

## Usage
### 1. Place Your Research Papers
Ensure that all PDFs are inside the `research_papers/` directory.

### 2. Run the Application
```bash
streamlit run main.py
```

### 3. Interact with the App
- Click **"Document Embedding"** to create vector embeddings.
- Enter your query in the text input field.
- The app will retrieve relevant information and generate answers.

## File Structure
```
📂 rag-groq-ollama
│── 📂 research_papers        # Folder for storing research paper PDFs
│── 📜 .env                   # Environment variables
│── 📜 requirements.txt       # Python dependencies
│── 📜 rag_hf.py                # Streamlit app embedding used by hf
│── 📜 rag_ollama.py      # embeddings are local
│── 📜 README.md            
```

## Future Enhancements
- Improve retrieval accuracy with better chunking techniques.
- Support multiple document formats (e.g., TXT, DOCX).
- Optimize response generation with hybrid retrieval models.

## Contributors
Developed by **Sheema Masood** | Powered by **Streamlit** 🚀

## License
This project is open-source and available under the **MIT License**.

