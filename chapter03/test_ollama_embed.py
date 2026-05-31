from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize embeddings
embedder = OllamaEmbeddings(
    model=os.getenv('OLLAMA_EMBED_MODEL'),
    base_url=os.getenv('OLLAMA_BASE_URL')
)

# Test embedding
text = "Test embedding generation"
embedding = embedder.embed_query(text)

print(f"Embedding generated successfully!")
print(f"Dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
