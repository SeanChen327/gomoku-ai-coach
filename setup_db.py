from google import genai
from pinecone import Pinecone
from google.genai.types import EmbedContentConfig

# 1. API Configuration
GEMINI_API_KEY = "AIzaSyCvpS6f_UrgAS2FhdD6ZEK32WrBbsxRONo" 
PINECONE_API_KEY = "pcsk_2tCAii_HNhEUG8tnrrw6bfrNxVFYa7A4eBVdjjZESMFyMwuRQ5o1vZXn1KsoGtuA3f6GEq"

# Initialize clients
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("tictactoe-rag")

# 2. English Knowledge Base
# We use professional terminology like "Radiating Influence" and "Counter-play"
knowledge_base = [
    {
        "id": "kb-1", 
        "text": "Board Feature: Player X occupies the center. Expert Advice: The center (index 4) is the most critical position. Player X now has radiating influence across all lines. The computer should prioritize occupying the four corners to defend effectively."
    },
    {
        "id": "kb-2", 
        "text": "Board Feature: Empty board, game start. Expert Advice: The best opening moves are either the center or any of the four corners, as they offer the highest number of potential winning lines."
    },
    {
        "id": "kb-3", 
        "text": "User Query: Computer difficulty, tactical intent, or algorithm. Expert Advice: On 'Hard' difficulty, the computer uses the Minimax algorithm. It calculates every possible outcome to ensure it never loses, making a draw the best possible result for the player."
    },
    {
        "id": "kb-4", 
        "text": "Board Feature: Player X is in a corner, Computer O is on an adjacent edge. Expert Advice: The player is likely setting up a 'Fork' (double-threat) trap. The computer must immediately seize another corner or the center to neutralize this threat."
    },
    # 【新增】：专门针对你遇到的这种开局情况的专家指导
    {
        "id": "kb-5",
        "text": "Board Feature: Player X occupies a corner, Computer O occupies the center. Expert Advice: Player X MUST NOT play the opposite corner, as the diagonal is already blocked by O. Instead, Player X should play an adjacent corner (like index 2 or 6) to maintain flexibility and attempt to set up a double-threat (Fork)."
    }
]

print("Starting to embed knowledge base and upload to Pinecone...")

vectors_to_upsert = []
for item in knowledge_base:
    # Generate embedding with fixed 768 dimensions
    embedding_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=item["text"],
        config=EmbedContentConfig(output_dimensionality=768)
    )
    vector_values = embedding_result.embeddings[0].values
    
    # Pack for Pinecone
    vectors_to_upsert.append(
        (item["id"], vector_values, {"text": item["text"]})
    )

# Upsert (Update/Insert) the data
index.upsert(vectors=vectors_to_upsert)
print("✅ Knowledge base successfully uploaded in English!")