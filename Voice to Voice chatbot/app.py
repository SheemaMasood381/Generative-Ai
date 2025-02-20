import gradio as gr
import whisper
from groq import Groq
from gtts import gTTS
import os

# Initialize Whisper model for transcription
model = whisper.load_model("base")

# Get API key from Hugging Face Spaces secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)




# Function to query the LLM using Groq API  (ye fx input receive krega or llm se trancribe krega)
def get_llm_response(input_text):
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": input_text,
        }],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content


# Function to convert text to speech using gTTS
def text_to_speech(text,output_audio="output_audio.mp3"):
    tts = gTTS(text)
    tts.save(output_audio)
    return output_audio


def chatbot(audio):
  result=model.transcribe(audio)
  user_text=result['text']
  response_text=get_llm_response(user_text)
  output_audio=text_to_speech(response_text)
  return response_text,output_audio


# Create the chatbot interface
with gr.Blocks() as iface:
    # Title & Description
    gr.Markdown("# 🎙️ AI Voice Chatbot")
    gr.Markdown("Speak into the microphone, and the AI will transcribe, process, and respond with both text and voice.")

    # Input Section
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="🎤 Speak", interactive=True, elem_id="box-style")

    # Output Section
    with gr.Row():
        text_output = gr.Textbox(label="💬 AI Response", interactive=False, elem_id="box-style")

    with gr.Row():
        audio_output = gr.Audio(type="filepath", label="🔊 AI Voice Response", elem_id="box-style")

    # Create a button for interaction
    submit_btn = gr.Button("🚀 Start Chat")
    submit_btn.click(fn=chatbot, inputs=audio_input, outputs=[text_output, audio_output])

    # Footer Section
    gr.Markdown("<hr>")  # Adds a line separator
    gr.Markdown("<p style='text-align: center; font-size: 14px;'>🌟 Developed by <b>Sheema Masood</b> | Built with <b>Gradio</b> 🌟</p>")


iface.launch()
