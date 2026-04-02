import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.genai.types import EmbedContentConfig
from pydantic import BaseModel
from typing import Optional, Any, Dict
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

# Secure environment variable access
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    logger.error("Critical API Keys are missing in .env file.")
    raise ValueError("API Keys are missing. Please set them in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("tictactoe-rag")

# --- UPDATED DATA MODELS (Hybrid Intelligence Architecture) ---

class ChatRequest(BaseModel):
    message: str
    board: list[str]
    last_evaluation: Optional[Dict[str, Any]] = None

class Move(BaseModel):
    step: int
    player: str
    index: int
    board_after: list[str]
    evaluation_label: str = ""
    comment: str = ""
    missed_best_move: Any = ""

class GameReportRequest(BaseModel):
    history: list[Move]
    final_result: str

# --- CORE LOGIC FUNCTIONS ---

def analyze_board(board: list[str]) -> str:
    win_patterns = [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], 
        [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], 
        [0, 5, 10, 15], [3, 6, 9, 12] 
    ]
    empty_spots = [str(i) for i, v in enumerate(board) if v == ""]
    
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
    center_indices = [5, 6, 9, 10]
    center_occupants = [board[i] for i in center_indices if board[i] != ""]
    center_status = "empty" if not center_occupants else f"occupied by {', '.join(set(center_occupants))}"
        
    search_query = f"4x4 Board center is {center_status}. User question: {user_message}"
    
    try:
        query_embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=search_query,
            config=EmbedContentConfig(output_dimensionality=768)
        )
        query_vector = query_embedding_result.embeddings[0].values
        
        search_results = pinecone_index.query(vector=query_vector, top_k=2, include_metadata=True)
        retrieved_texts = [match['metadata']['text'] for match in search_results['matches'] if match['score'] > 0.5]
        
        return "\n".join(retrieved_texts) if retrieved_texts else "No specific expert guidance found."
    except Exception as e:
        logger.warning(f"RAG Retrieval failed: {str(e)}")
        return "Expert guidance unavailable."

# --- API ENDPOINTS ---

@app.post("/api/chat")
def chat_with_ai(request: ChatRequest) -> dict:
    tactical_analysis = analyze_board(request.board)
    rag_context = retrieve_from_pinecone(request.board, request.message)
    
    move_context = ""
    if request.last_evaluation:
        le = request.last_evaluation
        move_context = f"Player X's last move (Index {le.get('index')}) was algorithmically evaluated as: [{le.get('evaluation_label')}]. Algorithm comment: '{le.get('comment')}'. "
        if le.get('missed_best_move') != "":
            move_context += f"They missed the optimal move at index {le.get('missed_best_move')}."

    system_prompt = f"""
    [ROLE]: You are a professional 4x4 Tic-Tac-Toe AI Coach. 
    Human plays 'X', Computer plays 'O'. Help 'X' defeat 'O'.
    
    [CURRENT BOARD STATE]: {request.board}
    [TACTICAL ANALYSIS]: {tactical_analysis}
    [RECENT MOVE CONTEXT]: {move_context}
    [RAG EXPERT CONTEXT]: {rag_context}
    
    User Question: "{request.message}"
    
    [STRICT GUIDELINES]:
    1. If the user asks about their last move, heavily rely on the [RECENT MOVE CONTEXT] to answer. Translate the algorithmic tag into friendly coaching advice.
    2. Suggest legal moves based on the Tactical Analysis.
    3. Keep response concise (under 80 words) and conversational.
    """
    
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=system_prompt)
        return {"reply": response.text}
    except Exception as e:
        logger.error(f"LLM Generation Error: {str(e)}")
        return {"reply": "The AI coach is temporarily unavailable."}

@app.post("/api/generate-report")
async def generate_report(request: GameReportRequest):
    history_summary = "\n".join([
        f"Step {m.step} ({m.player} to {m.index}): [{m.evaluation_label.upper()}] - {m.comment}" 
        for m in request.history
    ])
    
    report_prompt = f"""
    [TASK]: Provide a brief Executive Summary of this 4x4 Tic-Tac-Toe match.
    [FINAL RESULT]: {request.final_result}
    
    [TAGGED GAME HISTORY]:
    {history_summary}

    [INSTRUCTIONS]:
    The provided game history already contains algorithmic evaluations (e.g., MISSED_WIN, CRITICAL_MISTAKE, GOOD_MOVE).
    Do NOT recalculate the game. Simply summarize the data:
    1. Overall Assessment (1 sentence).
    2. Point out the most critical mistake Player X made (if any) based on the tags.
    3. Point out Player X's best move based on the tags.
    
    Format nicely with markdown. Limit to 100 words. Keep it professional.
    """

    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=report_prompt)
        return {
            "report_text": response.text,
            "raw_history": [m.dict() for m in request.history] 
        }
    except Exception as e:
        logger.error(f"Report Generation Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate match report.")