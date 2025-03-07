# Voice to Voice Chatbot

This project implements a Voice to Voice AI chatbot using advanced machine learning models. Users can interact with the chatbot by speaking into their microphone. The chatbot transcribes the speech, processes the text, generates a response, and converts the response back into speech.

## Features

- **Speech-to-Text**: Transcribe spoken words into text using Whisper.
- **Text Processing**: Use the Groq API with the Llama3-8B model to generate relevant responses.
- **Text-to-Speech**: Convert the AI-generated text response back into speech using Google Text-to-Speech (gTTS).
- **Interactive Web Interface**: Utilize Gradio to create an interactive web application for real-time voice-based interactions.

## Files

- **AiChatBot-a-Hugging-Face-Space-by-SheemaMasood.png**: An image file related to the project.
- **VOice toVoice WEb App INterface.png**: An image file related to the project.
- **voice_to_voice_GenAI_Application_using_Groq (1).ipynb**: A Jupyter Notebook demonstrating the implementation of the voice-to-voice chatbot using Whisper, Groq API, and gTTS.

## Installation

To run this application, you need to have Python installed. Follow the steps below to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/SheemaMasood381/Generative-Ai.git
    cd Generative-Ai/02_Voice to Voice chatbot
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

This will launch the Gradio interface, where you can speak into the microphone to interact with the AI chatbot.

## How It Works

1. **Speech Input**: The user speaks into the microphone.
2. **Transcription**: The Whisper model transcribes the speech into text.
3. **Text Processing**: The transcribed text is sent to the Llama3-8B model via the Groq API, which generates a response.
4. **Text-to-Speech**: The response is converted into speech using gTTS.
5. **Output**: The application outputs both the text response and an audio file for playback.

## Example

1. Speak into the microphone.
2. The AI transcribes your speech, generates a response, and plays the response as audio.

## Dependencies

The application relies on the following libraries:

- `gradio`
- `whisper`
- `groq`
- `gtts`

Make sure to install these dependencies using the provided `requirements.txt` file.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Whisper](https://github.com/openai/whisper) for speech-to-text transcription.
- [Groq](https://groq.com/) for advanced text processing.
- [gTTS](https://pypi.org/project/gTTS/) for text-to-speech conversion.
- [Gradio](https://gradio.app/) for the interactive web interface.

---

Developed by Sheema Masood 🚀
