from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is successfully loaded
if google_api_key is None:
    print("API key not found. Please make sure the GOOGLE_API_KEY is set in your .env file.")
else:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

    # Define prompt templates (no need for separate Runnable chains)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a facts expert who knows facts about {animal}."),
            ("human", "Tell me {fact_count} facts."),
        ]
    )

    # Create the combined chain using LangChain Expression Language (LCEL)
    chain = prompt_template | model | StrOutputParser()
    # chain = prompt_template | model

    # Run the chain
    result = chain.invoke({"animal": "elephant", "fact_count": 1})

    # Output
    print(result)
