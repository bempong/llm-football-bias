# Test the Hugging Face Inference API with the FinBERT model for sentiment analysis.

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

result = client.text_classification(
    "I like you. I love you",
    model="ProsusAI/finbert",
)

print("Result: ", result)