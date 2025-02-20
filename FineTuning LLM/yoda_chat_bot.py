import gradio as gr
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the base model and LoRA adapter
MODEL_PATH = "yoda_chat_model"
ADAPTER_PATH = "yoda_chat_adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Model loaded successfully!")

def generate_response(question):
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**tokens, max_new_tokens=50)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# LLM Judge System Prompt
system_prompt = """
You are an impartial judge that evaluates if text was written by Yoda.

An example piece of text from Yoda is:
"The very Republic is threatened, if involved the Sith are. Hard to see, the dark side is."

Now, analyze some new text carefully and respond on if it follows the same style of Yoda. Be critical to identify any issues in the text.
Then convert your feedback into a number between 0 and 10: 10 if the text is written exactly in Yoda's style, 5 if mixed faithfulness to the style, or 0 if the text is not at all written in Yoda's style.

The format of your response should be a JSON dictionary and nothing else:
{"score": <score between 0 and 10>}
"""

class LLMJudgeEvaluator:
    def __init__(self, judge_model):
        self.judge_model = judge_model
        self.prompt_template = "Evaluate this text: {text}"
    
    def score(self, text):
        prompt = self.prompt_template.format(text=text)
        response = """{"score": 8}"""  # Placeholder logic
        res_dict = json.loads(response)
        return res_dict["score"] / 10  # Normalize to [0,1]

# Mock judge model (replace with real judge model if available)
judge = LLMJudgeEvaluator(judge_model=None)

def yoda_chat_and_judge(prompt):
    response = generate_response(prompt)
    score = judge.score(response)
    result = f"\U0001F7E2 **Yoda Says:** {response}\n\n\U0001F535 **Yoda Style Score:** {score:.2f} (Closer to 1.0 = More Yoda-like!)"
    return result

# Launch Gradio UI
iface = gr.Interface(
    fn=yoda_chat_and_judge,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=gr.Textbox(label="Model Response & Score"),
    title="🟢 Yoda Chat & Judge",
    description="Chat with Yoda and see how accurate his style is!"
)

iface.launch()
