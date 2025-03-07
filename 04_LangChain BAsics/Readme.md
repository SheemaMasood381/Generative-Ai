# LangChain Basics

This project demonstrates the basics of using LangChain for creating chat-based applications. Specifically, it showcases how to build a chain of operations involving prompt templates and machine learning models for generating responses.

## Features

- **Prompt Templates**: Define structured prompts for the model to follow.
- **Machine Learning Integration**: Utilize the Google Generative AI model for generating responses.
- **LangChain**: Combine various components into a seamless chain of operations.

## Files

- **01_chain_basics.py**: A basic script demonstrating the creation and execution of a LangChain with prompt templates and the Google Generative AI model.
- **02_chain_fact.py**: An advanced script that extends the basic chain to generate facts about a specified animal.
- **requirements.txt**: List of dependencies required to run the scripts.
- **plots/chain.PNG**: An image file related to the project.
- **plots/Messages.PNG**: An image file related to the project.

## Installation

To run this application, you need to have Python installed. Follow the steps below to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/SheemaMasood381/Generative-Ai.git
    cd Generative-Ai/04_LangChain BAsics
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

### Basic Chain

To run the basic chain script, execute the following command:
```bash
python 01_chain_basics.py
```

### Fact Chain

To run the advanced fact chain script, execute the following command:
```bash
python 02_chain_fact.py
```

## How It Works

1. **Prompt Templates**: Define the structure of the prompts that will be used to generate responses.
2. **Model Initialization**: Initialize the Google Generative AI model with the provided API key.
3. **Chain Creation**: Combine the prompt templates and the model into a chain using LangChain.
4. **Execution**: Provide input parameters to the chain and execute it to get the desired output.

## Example

### Basic Chain
1. The script sets up a prompt template to ask for facts about an animal.
2. It initializes the Google Generative AI model.
3. The chain is created and executed with the input parameters.

### Fact Chain
1. Similar to the basic chain, but the script is designed to generate a specified number of facts about an animal.

## Dependencies

The application relies on the following libraries:

- `langchain`
- `langchain_google_genai`
- `tenacity`
- `google-generativeai`

Make sure to install these dependencies using the provided `requirements.txt` file.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [LangChain](https://langchain.com/) for providing the framework to create chains of operations.
- [Google Generative AI](https://ai.google/) for the machine learning model.
- [Gradio](https://gradio.app/) for the interactive web interface (if used).

---

Developed by Sheema Masood 🚀
