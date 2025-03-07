# Frameworks

This directory contains various frameworks and tools used for data ingestion, transformation, embeddings, and vector operations. Each sub-directory focuses on a specific aspect of data processing and machine learning.

## Sub-Directories

### 01_DataIngestion

- **Description**: This folder contains scripts and resources for data ingestion processes.
- **Files**:
  - `speech.txt`: A text file with content related to mental health institutions.
  - `attention.pdf`: A PDF file included for reference.

### 02_DataTransformation

- **Description**: This folder is dedicated to data transformation techniques and scripts.
- **Files**: [No specific files listed]

### 03_Embeddings

- **Description**: This folder focuses on embedding techniques using various models.
- **Files**:
  - `speech.txt`: A text file with content related to mental health institutions.
  - `2_Chroma.ipynb`: A Jupyter Notebook demonstrating the use of ChromaDB with LangChain and Hugging Face embeddings.

### 04_vectors

- **Description**: This folder contains resources and scripts for vector operations and similarity searches.
- **Files**:
  - `speech.txt`: A text file with content related to mental health institutions.
  - `1_Faiss.ipynb`: A Jupyter Notebook demonstrating the use of FAISS for similarity search.
  - `attention.pdf`: A PDF file included for reference.

### plots

- **Description**: This folder contains image files related to plots and visualizations.
- **Files**:
  - `chain.PNG`: An image file related to the project.
  - `Messages.PNG`: An image file related to the project.

## Installation

To run the scripts in these directories, you need to have Python installed. Follow the steps below to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/SheemaMasood381/Generative-Ai.git
    cd Generative-Ai/05_Frameworks
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

### Data Ingestion

To run the data ingestion script, execute the following command:
```bash
python 01_DataIngestion/speech.txt
```

### Data Transformation

To run the data transformation script, execute the following command:
```bash
python 02_DataTransformation/[script_name].py
```

### Embeddings

To run the embeddings script, execute the following command:
```bash
jupyter notebook 03_Embeddings/2_Chroma.ipynb
```

### Vectors

To run the vector operations script, execute the following command:
```bash
jupyter notebook 04_vectors/1_Faiss.ipynb
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [LangChain](https://langchain.com/) for providing the framework to create chains of operations.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.
- [ChromaDB](https://chroma.dev/) for vector database operations.
- [Hugging Face](https://huggingface.co/) for pre-trained models and embeddings.

---

Developed by Sheema Masood 🚀
