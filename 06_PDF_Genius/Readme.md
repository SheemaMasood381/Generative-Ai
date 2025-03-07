# PDF Genius - Chat & Summarizer




## 🚀 About PDF Genius

PDF Genius is an AI-powered application that allows users to chat with their PDF documents and generate summaries using Google's Gemini AI. This tool helps users extract key insights from PDFs efficiently.

## 🌟 Features

- 📄 **Upload multiple PDF files**
- 💬 **Chat with your PDFs** using an AI-powered Q&A system
- 📑 **Summarize documents** in short, medium, or detailed formats
- 💾 **Vector store for efficient retrieval** using ChromaDB
- ⚡ **Powered by Google Gemini AI for responses and embeddings**
- 🎨 **User-friendly interface built with Streamlit**

## 🛠️ Tech Stack

- **Streamlit** - Frontend UI
- **PyPDF2** - Extract text from PDFs
- **LangChain** - Text processing & retrieval
- **ChromaDB** - Vector storage for embedding-based search
- **Google Gemini AI** - Chat & Summarization
- **dotenv** - Environment variable management

## 📂 Repository Structure

```
📦 PDF Genius
├── 📄 app.py             # Main Streamlit application
├── 📄 chat.py            # Chat logic implementation
├── 📄 requirements.txt   # Python dependencies
├── 🖼️ ui (1).png         # UI Screenshot 1
├── 🖼️ ui (2).png         # UI Screenshot 2
└── 📂 chroma_db          # Vector database storage
```

## 🔧 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/pdf-genius.git
   cd pdf-genius
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up API Keys** (Google Gemini AI)
   - Create a `.env` file in the project root and add:
     ```
     GOOGLE_API_KEY=your_google_api_key
     ```
4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 🚀 Deployment on Hugging Face Spaces

PDF Genius is deployed on Hugging Face Spaces. You can access it [here](https://huggingface.co/spaces/SheemaMasood/PDFGenius).

## 👩‍💻 Developed By

**Sheema Masood**

---

🔗 Connect with me:  
[GitHub](https://github.com/sheemamasood381/) | [LinkedIn](https://www.linkedin.com/in/sheema-masood/)

