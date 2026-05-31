from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize LLM
llm = ChatOllama(
    model=os.getenv('OLLAMA_LLM_MODEL'),
    base_url=os.getenv('OLLAMA_BASE_URL')
)

# Test generation
response = llm.invoke("Say 'LLM working!'")
print(response.content)
