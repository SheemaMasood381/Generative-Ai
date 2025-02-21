import os
import gradio as gr
import torch
import json
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn.functional as F  # Softmax ke liye

# ✅ OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
assert OPENROUTER_API_KEY, "You must set your OpenRouter API key in Hugging Face Secrets!"

# ✅ Load Yoda Chat Model (Only Once)
MODEL_PATH = "yoda_chat_model"
ADAPTER_PATH = "yoda_chat_adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔄 Loading Yoda Model...")

base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("✅ Model Loaded Successfully!")

# ✅ System Prompt for OpenRouter Judge
SYSTEM_PROMPT = """
You are an impartial judge that evaluates if text was written in the style of Yoda.

An example piece of text from Yoda is:
"Do, or do not. There is no try."

Now, analyze some new text carefully and respond on if it follows the same style of Yoda.
Be critical to identify any issues in the text.

Then convert your feedback into a number between 0 and 10: 
10 if the text is written exactly in the style of Yoda, 
5 if mixed faithfulness to the style, 
0 if the text is not at all written in the style of Yoda.

The format of your response should be a JSON dictionary and nothing else:
{{"score": <score between 0 and 10>}}
"""

class LLMJudgeEvaluator:
    """Evaluates how 'Yoda-like' a text response is using OpenRouter API"""
    def __init__(self, model_name, api_key, system_prompt):
        self.model_name = model_name
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.prompt_template = "Evaluate this text: {text}"

    def ask(self, user_prompt):
        """Send a request to OpenRouter API and get response"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 10
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        
        try:
            json_response = response.json()
            res_text = json_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError):
            print(f"Error parsing OpenRouter response: {response.text}")
            res_text = '{"score": 0}'  # Default score 0 if error
        
        return res_text

    def score(self, text):
        """Evaluate response style and return score"""
        prompt = self.prompt_template.format(text=text)
        try:
            res = self.ask(prompt)
            res_dict = json.loads(res)  # Parse JSON response
            max_score = 10
            score = res_dict.get("score", 0) / max_score
            return max(0.0, min(score, 1.0))  # Clip score between 0-1
        except Exception as e:
            print(f"Error scoring text: {e}")
            return 0.0  # Default score if error

# ✅ Initialize the Judge Model using OpenRouter API
model_name = "liquid/lfm-40b"  # OpenRouter model
judge = LLMJudgeEvaluator(model_name, OPENROUTER_API_KEY, SYSTEM_PROMPT)

# ✅ Define the template globally
template_without_answer = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
template_with_answer = template_without_answer + "{answer}<end_of_turn>\n"


def chat(question, max_new_tokens=32, temperature=0.7, only_answer=False):
    # 1. Construct the prompt using the template
    prompt = template_without_answer.format(question=question)
    # 2. Tokenize the text
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    # 3. Feed through the model to predict the next token probabilities
    with torch.no_grad():
        outputs = model.generate(**input_ids, do_sample=True,
        max_new_tokens=max_new_tokens, temperature=temperature)
    output_tokens = outputs[0]
    if only_answer:
        output_tokens = output_tokens[input_ids['input_ids'].shape[1]:]

    # 5. Decode the tokens
    result = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return result


def yoda_chat_and_judge(question):
    """Generate Yoda-style response and evaluate it"""
    
    yield """
    <div style='font-family: Arial, sans-serif; padding: 20px; background: #222; border-radius: 10px; color: white; text-align: center;'>
        <h2 style='color: #f4d03f;'>🟢 Yoda Processing...</h2>
        <p style='font-size: 18px; background: #333; padding: 10px; border-radius: 5px; color: #ecf0f1;'>
            ⏳ "Patience, you must have, my young Padawan..." 🤔
        </p>
    </div>
    """
    
    # ✅ Use `chat()` function instead of `generate_response()`
    response = chat(question, only_answer=True) 

    # ✅ Evaluate response
    score = judge.score(response)

    # Progress bar color based on score
    color = "#e74c3c" if score < 0.4 else "#f1c40f" if score < 0.7 else "#2ecc71"

    result = f"""
    <div style='font-family: Arial, sans-serif; padding: 20px; background: #222; border-radius: 10px; color: white;'>
        <h2 style='color: #f4d03f;'>🟢 Yoda Says:</h2>
        <p style='font-size: 18px; background: #333; padding: 10px; border-radius: 5px; color: #ecf0f1; max-height: 200px; overflow-y: auto;'>💬 {response}</p>
        
        <h3 style='margin-top: 20px; color: #3498db;'>🔵 Yoda Style Score:</h3>
        <div style="background: #444; border-radius: 5px; padding: 5px; margin-top: 5px;">
            <div style="width: {score*100}%; background: {color}; height: 20px; border-radius: 5px;"></div>
        </div>
        <p style='font-size: 16px; color: {color}; font-weight: bold;'>Score: {score:.2f} (Closer to 1.0 = More Yoda-like!)</p>
    </div>
    """
    
    yield result  # ✅ Return final result after processing


with gr.Blocks(css="""
    .textbox, .output-box { height: 300px !important; overflow-y: auto; }
    .footer { 
        position: fixed; 
        bottom: 0; 
        width: 100%; 
        background: #222; 
        color: white; 
        padding: 10px; 
        text-align: center; 
        border-top: 2px solid #f4d03f; 
        font-size: 14px;
    }
""") as iface:
    gr.Markdown("# 🟢 Yoda Chat & Judge")
    gr.Markdown("Chat with Yoda and see how accurate his style is!")

    with gr.Row():
        inp = gr.Textbox(label="Enter your question", placeholder="Type something like Yoda would...", lines=10, elem_id="textbox")
        out = gr.HTML(value="💬 Waiting for input...", elem_id="output-box")  # ✅ Default output box remains visible

    btn = gr.Button("Generate")
    btn.click(yoda_chat_and_judge, inputs=inp, outputs=out)

    # ✅ Fixed Footer
    # ✅ Fixed Footer with Clickable Links
    gr.HTML(
        """
        <div class='footer' style="text-align: center; padding: 10px; font-size: 16px;">
            🌟 Developed by <b>Sheema Masood</b> | May the Force Be With You! 🌌 <br>
            🔗 Connect with me: 
            <a href="https://github.com/SheemaMasood381" target="_blank" style="color: #f4d03f; text-decoration: none;">GitHub</a> |
            <a href="https://www.linkedin.com/in/sheema-masood/" target="_blank" style="color: #3498db; text-decoration: none;">LinkedIn</a> |
            <a href="https://www.kaggle.com/sheemamasood" target="_blank" style="color: #e74c3c; text-decoration: none;">Kaggle</a>
        </div>
        """
    )



iface.launch()
