import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from google import genai 
from google.genai.types import EmbedContentConfig
from pinecone import Pinecone
from dotenv import load_dotenv
from passlib.context import CryptContext
from jose import JWTError, jwt

# --- SQLAlchemy Imports ---
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# [SECURITY UPDATE] Initialize logging for secure auditing and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- DATABASE CONFIGURATION ---
# Dynamically switch between SQLite (local) and PostgreSQL (Render)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tictactoe.db")

# Fix for SQLAlchemy 1.4+ which dropped support for 'postgres://' URI scheme
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DB MODELS (ORM) ---
class UserORM(Base):
    """SQLAlchemy model for the 'users' table."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

# Create tables in the database
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Note: For strict production, allowed origins should be limited to the exact Render domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SECURITY & AUTHENTICATION SETTINGS ---
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_default_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

# Secure environment variable access
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    logger.error("Critical API Keys are missing in .env file.")
    raise ValueError("API Keys are missing. Please set them in your environment.")

client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("tictactoe-rag")


# --- DATA MODELS (Pydantic) ---

class Token(BaseModel):
    """Token model for authentication responses."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Data extracted from the JWT token."""
    username: Optional[str] = None

class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str
    password: str
    email: EmailStr

class UserOut(BaseModel):
    """Schema for returning user data securely (excluding password)."""
    username: str
    email: str
    disabled: bool
    
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    """Request schema for AI Chat."""
    message: str
    board: list[str]
    last_evaluation: Optional[Dict[str, Any]] = None

class Move(BaseModel):
    """Schema representing a single game move."""
    step: int
    player: str
    index: int
    board_after: list[str]
    evaluation_label: str = ""
    comment: str = ""
    missed_best_move: Any = ""

class GameReportRequest(BaseModel):
    """Request schema for generating an AI match report."""
    history: list[Move]
    final_result: str


# --- UTILITIES & DEPENDENCIES ---

def get_db():
    """Dependency to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain-text password using bcrypt."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserORM:
    """Validates the provided JWT token and returns the corresponding user from DB."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
        
    user = db.query(UserORM).filter(UserORM.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserORM = Depends(get_current_user)) -> UserORM:
    """Ensures the current authenticated user is active."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# --- CORE LOGIC FUNCTIONS ---

def analyze_board(board: list[str]) -> str:
    """Analyzes the 15x15 Gomoku board for basic tactical threats."""
    # This prevents the LLM from trying to parse 225 cells blindly.
    # We do a quick heuristic scan for any sequences of 4.
    BOARD_SIZE = 15
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            idx = r * BOARD_SIZE + c
            player = board[idx]
            if not player:
                continue
                
            for dr, dc in directions:
                count = 1
                for step in range(1, 4):
                    nr, nc = r + dr * step, c + dc * step
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr * BOARD_SIZE + nc] == player:
                            count += 1
                        else:
                            break
                    else:
                        break
                
                if count == 4:
                    if player == "O":
                        return f"[CRITICAL]: Computer (O) has 4 in a row starting near index {idx}! Block immediately."
                    if player == "X":
                        return f"[OPPORTUNITY]: Human (X) has 4 in a row near index {idx}! Push for the win."
                        
    return "[NEUTRAL]: No immediate 4-in-a-row threats detected. Focus on building open threes and controlling intersections."

def retrieve_from_pinecone(board: list[str], user_message: str) -> str:
    """Retrieves expert Gomoku strategies from Pinecone vector DB."""
    # (Note: For optimal production behavior, you should eventually update setup_db.py 
    #  to push Gomoku logic instead of 4x4 Tic-Tac-Toe logic into Pinecone).
    search_query = f"Gomoku 15x15 Strategy. User question: {user_message}"
    
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

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health_check():
    """Lightweight health check endpoint for Keep-Alive Ping (Strategy B)."""
    return {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "AI Coach is awake and ready."
    }

@app.get("/")
def read_root():
    """Serves the frontend application interface."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "API is running, but index.html was not found."}

@app.post("/api/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Registers a new user and persists data to PostgreSQL/SQLite."""
    db_user = db.query(UserORM).filter(UserORM.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_email = db.query(UserORM).filter(UserORM.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_pwd = get_password_hash(user.password)
    new_user = UserORM(
        username=user.username, 
        email=user.email, 
        hashed_password=hashed_pwd
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.info(f"[SECURITY EVENT] New user registered successfully: {user.username}")
    return new_user

@app.post("/api/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticates a user against the database and returns a JWT access token."""
    user = db.query(UserORM).filter(UserORM.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/users/me", response_model=UserOut)
async def read_users_me(current_user: UserORM = Depends(get_current_active_user)):
    """Returns the currently authenticated user's profile."""
    return current_user

@app.post("/api/chat")
def chat_with_ai(request: ChatRequest, current_user: UserORM = Depends(get_current_active_user)) -> dict:
    """Handles user queries to the AI Coach using tactical analysis and RAG."""
    logger.info(f"[SECURITY EVENT] User '{current_user.username}' requested AI chat.")
    
    tactical_analysis = analyze_board(request.board)
    rag_context = retrieve_from_pinecone(request.board, request.message)
    
    move_context = ""
    if request.last_evaluation:
        le = request.last_evaluation
        move_context = f"Player X's last move (Index {le.get('index')}) was algorithmically evaluated as: [{le.get('evaluation_label')}]. Algorithm comment: '{le.get('comment')}'. "

    system_prompt = f"""
    [ROLE]: You are a professional 15x15 Gomoku (Five-in-a-Row) AI Coach. 
    Human plays Black stones ('X'), Computer plays White stones ('O').
    
    [TACTICAL ANALYSIS]: {tactical_analysis}
    [RECENT MOVE CONTEXT]: {move_context}
    [RAG EXPERT CONTEXT]: {rag_context}
    
    User Question: "{request.message}"
    
    [STRICT GUIDELINES]:
    1. Acknowledge this is Gomoku (15x15 grid), where the goal is 5 stones in a row.
    2. Do NOT attempt to read the entire 225-cell board blindly. Rely strictly on the [TACTICAL ANALYSIS] and [RECENT MOVE CONTEXT] provided above.
    3. Keep response concise (under 80 words), encouraging, and conversational.
    """
    
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=system_prompt)
        return {"reply": response.text}
    except Exception as e:
        logger.error(f"LLM Generation Error: {str(e)}")
        return {"reply": "The AI coach is temporarily unavailable."}

@app.post("/api/generate-report")
async def generate_report(request: GameReportRequest, current_user: UserORM = Depends(get_current_active_user)):
    """Generates an executive summary of the game using an LLM."""
    logger.info(f"[SECURITY EVENT] User '{current_user.username}' requested match report generation.")
    
    history_summary = "\n".join([
        f"Step {m.step} ({m.player} to {m.index}): [{m.evaluation_label.upper()}] - {m.comment}" 
        for m in request.history
    ])
    
    report_prompt = f"""
    [TASK]: Provide a brief Executive Summary of this 15x15 Gomoku match.
    [FINAL RESULT]: {request.final_result}
    
    [TAGGED GAME HISTORY (Sampled)]:
    {history_summary[-1000:]} # Sending only recent history to avoid token limits

    [INSTRUCTIONS]:
    1. Overall Assessment (1-2 sentences).
    2. Provide general advice for improving Gomoku gameplay (e.g., focus on open 3s and 4s).
    
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