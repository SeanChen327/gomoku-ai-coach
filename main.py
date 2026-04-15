import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.genai.types import EmbedContentConfig
from pydantic import BaseModel
from google import genai 
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# [SECURITY UPDATE] Initialize logging for secure auditing and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use os.getenv to securely read environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Security check to prevent application from running without critical credentials
if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API Keys are missing. Please set them in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("tictactoe-rag")

class ChatRequest(BaseModel):
    """
    Data model for incoming chat requests from the client.

    Attributes:
        message (str): The user's query or instruction to the AI coach.
        board (list[str]): The current state of the Tic-Tac-Toe board represented 
                           as a list of 16 strings (e.g., ["X", "O", "", ...]).
    """
    message: str
    board: list[str]

def analyze_board(board: list[str]) -> str:
    """
    Analyzes the current 4x4 Tic-Tac-Toe board and provides tactical advice.

    This function identifies immediate winning opportunities or critical threats
    that need to be blocked by checking all 10 possible winning lines in a 4x4 grid.

    Args:
        board (list[str]): A list of 16 strings representing the 4x4 board state.

    Returns:
        str: A formatted string containing tactical advice (CRITICAL, OPPORTUNITY, or NEUTRAL)
             with a list of recommended safe moves.
    """
    # 4x4 Winning Patterns
    win_patterns = [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], # Rows
        [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], # Cols
        [0, 5, 10, 15], [3, 6, 9, 12] # Diagonals
    ]
    empty_spots = [str(i) for i, v in enumerate(board) if v == ""]
    
    # Analyze critical lines (3 of the same, 1 empty)
    for p in win_patterns:
        line = [board[p[0]], board[p[1]], board[p[2]], board[p[3]]]
        if line.count("O") == 3 and line.count("") == 1:
            critical_index = p[line.index("")]
            return f"[CRITICAL]: Block O at {critical_index} immediately!"
        if line.count("X") == 3 and line.count("") == 1:
            win_index = p[line.index("")]
            return f"[OPPORTUNITY]: Play at {win_index} to win!"
            
    return f"[NEUTRAL]: No immediate threats. Safe moves available: {', '.join(empty_spots)}."

def retrieve_from_pinecone(board: list[str], user_message: str) -> str:
    """
    Retrieves expert Tic-Tac-Toe advice from the Pinecone vector database using RAG.

    Args:
        board (list[str]): The current state of the Tic-Tac-Toe board (length 16).
        user_message (str): The user's query or prompt.

    Returns:
        str: Aggregated context retrieved from the database, or a fallback string
             if no relevant data is found.
    """
    # In a 4x4, the center is the 4 inner squares: 5, 6, 9, 10.
    center_indices = [5, 6, 9, 10]
    center_occupants = [board[i] for i in center_indices if board[i] != ""]
    
    if not center_occupants:
        center_status = "empty"
    else:
        center_status = f"occupied by {', '.join(set(center_occupants))}"
        
    search_query = f"4x4 Board center is {center_status}. User question: {user_message}"
    
    query_embedding_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=search_query,
        config=EmbedContentConfig(output_dimensionality=768)
    )
    query_vector = query_embedding_result.embeddings[0].values
    
    search_results = pinecone_index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True
    )
    
    retrieved_texts = []
    for match in search_results['matches']:
        if match['score'] > 0.5: 
            retrieved_texts.append(match['metadata']['text'])
            
    if not retrieved_texts:
        return "No specific expert guidance found in the database."
        
    return "\n".join(retrieved_texts)

@app.post("/api/chat")
def chat_with_ai(request: ChatRequest) -> dict:
    """
    API endpoint to interact with the 4x4 Tic-Tac-Toe AI Coach.

    Accepts the current board state and a user message, augments the prompt with 
    tactical analysis and RAG context, and returns the LLM's coaching response.

    Args:
        request (ChatRequest): The validated request payload.

    Returns:
        dict: A dictionary containing the AI's reply, or a safe error message if
              the generation fails.
    """
    user_message = request.message
    current_board = request.board
    
    tactical_analysis = analyze_board(current_board)
    rag_context = retrieve_from_pinecone(current_board, user_message)
    
    # English System Prompt
    system_prompt = f"""
    [ROLE]: You are a professional 4x4 Tic-Tac-Toe AI Coach. Your ONLY client is the human player.
    Human plays 'X', Computer plays 'O'. Your mission is to help 'X' defeat 'O'.
    NEVER take the perspective of the computer ('O').

    Current 4x4 board array (indices 0-15): {current_board}
    [TACTICAL ANALYSIS (HIGHEST PRIORITY)]: {tactical_analysis}
    [RAG EXPERT CONTEXT]: {rag_context}
    
    Please answer the user's question: "{user_message}"
    
    [STRICT GUIDELINES]:
    1. All tactical advice MUST be from the perspective of player 'X'.
    2. Any move suggestion MUST be chosen from the "legal moves" provided in the tactical analysis.
    3. DO NOT suggest positions that are already occupied by 'X' or 'O'.
    4. Keep your answer professional, encouraging, and concise (under 100 words).
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt
        )
        return {"reply": response.text}
    except Exception as e:
        # [SECURITY UPDATE] Log the actual error internally, return generic message to user
        logger.error(f"LLM Generation Error: {str(e)}", exc_info=True)
        return {"reply": "An internal server error occurred while consulting the AI. Please try again later."}