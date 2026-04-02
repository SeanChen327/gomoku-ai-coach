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
                           as a list of 9 strings (e.g., ["X", "O", "", ...]).
    """
    message: str
    board: list[str]

def analyze_board(board: list[str]) -> str:
    """
    Analyzes the current Tic-Tac-Toe board and provides tactical advice.

    This function identifies immediate winning opportunities or critical threats
    that need to be blocked. It also filters out unsafe diagonal moves.

    Args:
        board (list[str]): A list of 9 strings representing the board state.

    Returns:
        str: A formatted string containing tactical advice (CRITICAL, OPPORTUNITY, or NEUTRAL)
             with a list of recommended safe moves.
    """
    win_patterns = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]
    empty_spots = [i for i, v in enumerate(board) if v == ""]
    
    # Diagonal Block Logic
    # If Player is at 0, and Computer is at 4, position 8 is USELESS.
    forbidden_spots = []
    if board[0] == "X" and board[4] == "O": forbidden_spots.append(8)
    if board[2] == "X" and board[4] == "O": forbidden_spots.append(6)
    if board[6] == "X" and board[4] == "O": forbidden_spots.append(2)
    if board[8] == "X" and board[4] == "O": forbidden_spots.append(0)
    
    # Final safe moves for the AI to recommend
    safe_moves = [str(i) for i in empty_spots if i not in forbidden_spots]

    for p in win_patterns:
        line = [board[p[0]], board[p[1]], board[p[2]]]
        if line.count("O") == 2 and line.count("") == 1:
            return f"[CRITICAL]: Block O at {p[line.index('')]}!"
        if line.count("X") == 2 and line.count("") == 1:
            return f"[OPPORTUNITY]: Win at {p[line.index('')]}!"
            
    return f"[NEUTRAL]: No immediate threats. STRONGLY RECOMMENDED moves: {', '.join(safe_moves)}. (Avoid blocked diagonals!)"

def retrieve_from_pinecone(board: list[str], user_message: str) -> str:
    """
    Retrieves expert Tic-Tac-Toe advice from the Pinecone vector database using RAG.

    Args:
        board (list[str]): The current state of the Tic-Tac-Toe board.
        user_message (str): The user's query or prompt.

    Returns:
        str: Aggregated context retrieved from the database, or a fallback string
             if no relevant data is found.
    """
    center_status = board[4] if board[4] else "empty"
    search_query = f"Board center is {center_status}. User question: {user_message}"
    
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
    API endpoint to interact with the Tic-Tac-Toe AI Coach.

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
    [ROLE]: You are a professional Tic-Tac-Toe AI Coach. Your ONLY client is the human player.
    Human plays 'X', Computer plays 'O'. Your mission is to help 'X' defeat 'O'.
    NEVER take the perspective of the computer ('O').

    Current board array (indices 0-8): {current_board}
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