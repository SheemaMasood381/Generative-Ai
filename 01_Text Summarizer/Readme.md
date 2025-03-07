# Text Summarizer

This project is a Text Summarization application that leverages a pre-trained Transformer model to summarize text. Users can either enter text directly or upload files (PDF, DOCX, or TXT) to generate concise summaries.

## Features

- Summarize text using a pre-trained Transformer model from Hugging Face.
- Extract text from PDF, DOCX, and TXT files.
- Interactive web interface powered by Gradio.

## Files

- **TextSummarizer-a-Hugging-Face-Space-by-SheemaMasood.png:** An image file related to the project.
- **Text_Summarization with transformers - Copy.ipynb:** A Jupyter Notebook demonstrating text summarization using transformers.
- **app.py:** The main application script that sets up the Gradio interface for text summarization.
- **requirements.txt:** List of dependencies required to run the application.

## Installation

To run this application, you need to have Python installed. Follow the steps below to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/SheemaMasood381/Generative-Ai.git
    cd Generative-Ai/01_Text Summarizer
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, execute the following command:
```bash
python app.py
```

This will launch the Gradio interface, where you can enter text or upload a file to generate a summary.

## How It Works

1. **Text Extraction:** The `extract_text` function reads the content of the uploaded file (PDF, DOCX, or TXT) and extracts the text.
2. **Text Summarization:** The `summarize_text` function uses the pre-trained Transformer model to generate a summary of the extracted or entered text.
3. **Gradio Interface:** The Gradio interface provides an interactive web UI for users to input text or upload files and view the generated summaries.

## Example

1. Enter text in the textbox or upload a file.
2. Click the "Submit" button to generate a summary.
3. The summarized text will be displayed in the output area.

## Dependencies

The application relies on the following libraries:

- `gradio`
- `transformers`
- `pdfplumber`
- `python-docx`
- `torch`
- `torchvision`
- `torchaudio`

Make sure to install these dependencies using the provided `requirements.txt` file.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the pre-trained Transformer models.
- [Gradio](https://gradio.app/) for the interactive web interface.

---

Developed by Sheema Masood 🚀
