from google import genai
from pinecone import Pinecone
from google.genai.types import EmbedContentConfig
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API Keys are missing. Please set them in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("tictactoe-rag")

# 4x4 Specialized Knowledge Base
knowledge_base = [
    {
        "id": "kb-4x4-1", 
        "text": "Board Feature: 4x4 Grid. Expert Advice: In a 4x4 game, the center consists of indices 5, 6, 10, and 11. Controlling these four inner squares is more strategically important than a single center point. They offer the most branching winning lines."
    },
    {
        "id": "kb-4x4-2", 
        "text": "Board Feature: 4x4 Diagonals. Expert Advice: Main diagonal is [0, 5, 10, 15] and anti-diagonal is [3, 6, 9, 12]. Winning requires four in a row. A three-in-a-row without a block is a 'threat' but not an immediate win."
    },
    {
        "id": "kb-4x4-3", 
        "text": "User Query: How to win 4x4? Expert Advice: Focus on creating 'Forks' where two lines of three intersect. Since the board is larger, players have more room to maneuver, making edge-to-corner transitions very effective."
    }
]

print("Updating Pinecone with 4x4 Expert Knowledge...")

vectors_to_upsert = []
for item in knowledge_base:
    embedding_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=item["text"],
        config=EmbedContentConfig(output_dimensionality=768)
    )
    vector_values = embedding_result.embeddings[0].values
    vectors_to_upsert.append((item["id"], vector_values, {"text": item["text"]}))

index.upsert(vectors=vectors_to_upsert)
print("✅ 4x4 Knowledge base successfully updated!")