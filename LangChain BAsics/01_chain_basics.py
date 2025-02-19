from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded correctly
if google_api_key is None:
    print("Error: GOOGLE_API_KEY not found. Please make sure the .env file is present and contains the key.")
    exit()  # Stop execution if the key is missing

# Initialize the model with the API key
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

# Create the combined chain
chain = prompt_template | model | StrOutputParser()

# Define input parameters
inputs = {"animal": "elephant", "fact_count": 1}

# Run the chain and get the result
try:
    result = chain.invoke(inputs)
    print(result)  # Print the result
except Exception as e:
    print(f"Error: {e}")
