import gradio as gr
from transformers import pipeline
import pdfplumber
from docx import Document

# Load the summarization model
summarizer = pipeline("summarization")

# Function to extract text from files
def extract_text(file_path):
    if file_path is None:
        return "No file uploaded."
    
    file_name = file_path.name.lower()

    try:
        if file_name.endswith(".pdf"):
            with pdfplumber.open(file_path.name) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif file_name.endswith(".docx"):
            doc = Document(file_path.name)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_name.endswith(".txt"):
            text = file_path.read().decode("utf-8")
        else:
            return "Unsupported file format. Please upload a PDF, DOCX, or TXT file."
        
        return text if text.strip() else "No readable text found in the file."

    except Exception as e:
        return f"Error reading file: {str(e)}"

# Function to summarize text
def summarize_text(text, file):
    if file is not None:
        text = extract_text(file)
    
    if text.strip() and "Error" not in text:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=True)
        return summary[0]["summary_text"]
    else:
        return "No valid text found to summarize."

# Gradio Interface
app = gr.Interface(
    fn=summarize_text, 
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter text here (or upload a file)..."), 
        gr.File(label="Upload a file (PDF, DOCX, TXT)")
    ], 
    outputs="text",
    title="📝 Text Summarization App",
    description="📄 Upload a document or enter text to summarize it using an AI-powered Transformer model.",
    theme="compact",
    allow_flagging="never",
    live=True
)

# Add footer
app.launch(share=True, debug=True)
print("\nDeveloped by Sheema Masood 🚀")
