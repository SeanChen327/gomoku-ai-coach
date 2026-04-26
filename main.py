# main.py
import logging
import os
import json
import asyncio
import uuid  
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict # [FIX] 引入 ConfigDict
from google import genai 
from dotenv import load_dotenv
from passlib.context import CryptContext
from jose import JWTError, jwt
from ai_governance import GomokuAIGovernance 
import time 

# --- Reliability Imports (Tenacity) ---
from tenacity import retry, stop_after_attempt, wait_exponential

# --- SQLAlchemy Imports ---
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, declarative_base # [FIX] 从 orm 导入 declarative_base
from sqlalchemy.sql import func

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 引入离线对弈引擎 ---
from ai_battle_engine import GomokuSimulator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
MOCK_AI = os.getenv("MOCK_AI", "false").lower() == "true"

# --- DATABASE CONFIGURATION ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tictactoe.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base() # [FIX] Warning resolved

# --- DB MODELS ---
class UserORM(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

class ScheduledMatchORM(Base):
    __tablename__ = "scheduled_matches"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="PENDING")  
    scheduled_time = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    match_data = Column(JSON, nullable=True) 
    final_result = Column(String, nullable=True)
    is_notified = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

app = FastAPI()
governance = GomokuAIGovernance()
CRON_SECRET = os.getenv("CRON_SECRET", "default_internal_secret_change_me")

# [SECURITY] Strict CORS Policy
# Replace the render.com URL with your actual production frontend URL if different.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://tictactoe-preview-service.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # Explicitly define allowed methods
    allow_headers=["Authorization", "Content-Type", "X-Cron-Secret"], # Explicitly define allowed headers
)

# --- SECURITY & AUTHENTICATION ---
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_default_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    logger.error("Critical API Keys are missing in .env file.")
    if not MOCK_AI:
        raise ValueError("API Keys are missing.")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# --- DATA MODELS (Pydantic) ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    password: str
    email: EmailStr

class UserOut(BaseModel):
    username: str
    email: str
    disabled: bool
    model_config = ConfigDict(from_attributes=True) # [FIX] Warning resolved

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    board: List[str] = Field(..., description="15x15 board array")
    last_evaluation: Optional[Dict[str, Any]] = None

    @field_validator('board')
    @classmethod
    def validate_board_size(cls, v):
        if len(v) != 225:
            raise ValueError("Board must be exactly 225 cells for 15x15 Gomoku.")
        if any(cell not in ["X", "O", ""] for cell in v):
            raise ValueError("Board cells must contain only 'X', 'O', or empty strings.")
        return v

class ChatResponse(BaseModel):
    reply: str
    telemetry: Dict[str, Any]
    trace_id: str

class Move(BaseModel):
    step: int
    player: str
    index: int
    board_after: List[str]
    evaluation_label: str = ""
    comment: str = ""
    missed_best_move: Any = ""

class GameReportRequest(BaseModel):
    history: List[Move]
    final_result: str

class MatchScheduleRequest(BaseModel):
    scheduled_time: datetime

# --- UTILITIES ---
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError: raise credentials_exception
    user = db.query(UserORM).filter(UserORM.username == username).first()
    if user is None: raise credentials_exception
    return user

async def get_current_active_user(current_user: UserORM = Depends(get_current_user)):
    if current_user.disabled: raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def analyze_board_v2(board: List[str]) -> str:
    simulator = GomokuSimulator()
    simulator.board = list(board) 
    best_move_idx = simulator.get_best_move(board, "X")
    if best_move_idx is None:
        return "[NEUTRAL]: Game over or board full."
    
    threat_score_to_o = simulator.evaluate_cell(board, best_move_idx, "X")
    coord = simulator.index_to_coord(best_move_idx)
    
    if threat_score_to_o >= 10000:
        return f"[CRITICAL]: Human (X) has a lethal threat near {coord}! AI (O) must block."
    elif threat_score_to_o >= 1000:
        return f"[WARNING]: Strategic advantage for Human (X) at {coord}."
    return f"[NEUTRAL]: Positional play continues. Key intersection: {coord}."

# --- [RELIABILITY & RAG UPGRADE] LANGCHAIN RAG SERVICE ---
class GomokuRagService:
    def __init__(self):
        if MOCK_AI and not GEMINI_API_KEY:
            return
            
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", output_dimensionality=768)
        self.vectorstore = PineconeVectorStore(index_name="tictactoe-rag", embedding=self.embeddings, pinecone_api_key=PINECONE_API_KEY)
        
        # [Feedback 3 Fix: RAG Tuning] 
        # Rationale for k=3: Experiments show k=1 lacks sufficient tactical diversity, 
        # while k>=5 introduces context noise and increases token latency/cost. 
        # k=3 is the optimal baseline. Next architectural step: Fetch k=10 and apply Cross-Encoder Re-ranking.
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

    async def _safe_retrieve(self, user_message: str, tactical_analysis: str, trace_id: str) -> str:
        """
        [Fallback & Context Strategy] Pinecone 检索增加超时限制，并融合棋盘上下文。
        """
        try:
            # [Feedback 3 Fix] 融合上下文：不仅搜用户的提问，还带入当前的战术状态
            enhanced_query = f"User asks: '{user_message}'. Current tactical situation: '{tactical_analysis}'"
            docs = await asyncio.wait_for(self.retriever.ainvoke(enhanced_query), timeout=3.0)
            return "\n".join([d.page_content for d in docs])
        except asyncio.TimeoutError:
            logger.warning(f"[{trace_id}] [Degradation] Pinecone retrieval timed out. Skipping RAG.")
            return "No external knowledge base available. Rely entirely on tactical analysis."
        except Exception as e:
            logger.warning(f"[{trace_id}] [Degradation] Pinecone error: {str(e)}. Skipping RAG.")
            return "No external knowledge base available. Rely entirely on tactical analysis."

    def _format_move_context(self, le: Dict[str, Any]) -> str:
        le = le or {}
        if not le: return "No recent move data available."
        ctx = f"Player X's last move (Index {le.get('index', 'N/A')}) was: [{le.get('evaluation_label')}]. Comment: '{le.get('comment')}'. "
        if le.get('suggested_next_move'): ctx += f"Suggested next tactical move: {le.get('suggested_next_move')}. "
        if le.get('win_rate'): ctx += f"Current win rate: {le.get('win_rate')}. "
        return ctx

    # [Retry Strategy] 遇到网络波动，最多重试 3 次，等待时间指数级增长
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_response(self, message: str, board: List[str], last_eval: Dict[str, Any], trace_id: str) -> str:
        tactical_analysis = analyze_board_v2(board)
        move_context = self._format_move_context(last_eval)
        
        # 传递战术分析结果给检索器
        rag_context = await self._safe_retrieve(message, tactical_analysis, trace_id)

        prompt = ChatPromptTemplate.from_template("""
        [ROLE]: You are a professional 15x15 Gomoku (Five-in-a-Row) AI Coach. 
        Human plays Black stones ('X'), Computer plays White stones ('O').
        
        [TACTICAL ANALYSIS]: {tactical_analysis}
        [RECENT MOVE CONTEXT]: {move_context}
        [RAG EXPERT CONTEXT]: {rag_context}
        
        User Question: "{message}"
        
        [STRICT GUIDELINES]:
        1. BOARD LIMITS: 15x15 grid. Columns A-O, Rows 1-15.
        2. NO HALLUCINATION: Only suggest exact coordinates provided in the TACTICAL ANALYSIS or RECENT MOVE.
        3. SYMBOL SEPARATION: Never combine player symbol with numbers (e.g., "X15").
        4. WIN RATE MANDATORY: You MUST explicitly state the user's current win rate percentage (e.g., "XX%") as provided in the [RECENT MOVE CONTEXT].
        5. Keep your response highly encouraging, conversational, and strictly under 80 words.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        
        # [Timeout Strategy] Gemini 调用 8 秒极限熔断
        return await asyncio.wait_for(
            chain.ainvoke({
                "tactical_analysis": tactical_analysis,
                "move_context": move_context,
                "rag_context": rag_context,
                "message": message
            }), 
            timeout=8.0
        )

rag_service = GomokuRagService()

# --- API ENDPOINTS ---

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "online", "timestamp": datetime.utcnow().isoformat()}

@app.get("/")
def read_root():
    if os.path.exists("index.html"): return FileResponse("index.html")
    return {"message": "API is running."}

@app.post("/api/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(UserORM).filter(UserORM.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    new_user = UserORM(username=user.username, email=user.email, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/api/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserORM).filter(UserORM.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    return {"access_token": create_access_token(data={"sub": user.username}), "token_type": "bearer"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest, current_user: UserORM = Depends(get_current_active_user)):
    trace_id = str(uuid.uuid4())
    logger.info(f"[{trace_id}] User '{current_user.username}' requested AI chat.")
    start_t = time.time()
    
    if governance.detect_adversarial_input(request.message):
        logger.warning(f"[{trace_id}] [SECURITY] Adversarial prompt blocked.")
        return ChatResponse(reply="I can only discuss Gomoku strategies.", telemetry={}, trace_id=trace_id)

    if MOCK_AI:
        await asyncio.sleep(1) 
        reply_text = "You're doing great with a 55% win rate! Tactically, playing at H9 is your best move right now. Keep it up!"
    else:
        try:
            # [修改] 现在调用解耦后的高可靠性服务方法
            reply_text = await rag_service.generate_response(
                message=request.message, 
                board=request.board, 
                last_eval=request.last_evaluation,
                trace_id=trace_id
            )
        except asyncio.TimeoutError:
            logger.error(f"[{trace_id}] Chat Error: Request timed out after retries.")
            raise HTTPException(status_code=504, detail="The AI coach took too long to analyze. Please try again.")
        except Exception as e:
            logger.error(f"[{trace_id}] Chat Error: {str(e)}")
            raise HTTPException(status_code=503, detail="The AI coach is temporarily unavailable. Please try again.")

    is_safe, reason = governance.validate_output_safety(reply_text, request.board)
    if not is_safe:
        logger.error(f"[{trace_id}] [Gr] Response suppressed: {reason}")
        reply_text = "I apologize, my analysis encountered an error. Let's look at the board again."

    metrics = governance.track_telemetry(start_t, reply_text)
    quality_score = governance.evaluate_response_consistency(reply_text, request.last_evaluation or {})

    if governance.requires_human_oversight(quality_score):
        logger.critical(f"[{trace_id}] [Hl] ESCALATE TO HITL. Score: {quality_score:.2f}, Reply: {reply_text}")

    logger.info(f"[{trace_id}] [PROD_LOG] User: {current_user.username} | Quality: {quality_score:.2f} | Metrics: {metrics}")
    return ChatResponse(reply=reply_text, telemetry=metrics, trace_id=trace_id)

@app.post("/api/generate-report")
async def generate_report(request: GameReportRequest, current_user: UserORM = Depends(get_current_active_user)):
    if MOCK_AI:
        await asyncio.sleep(1)
        return {
            "report_text": "[MOCK AI] This is a mocked executive summary. Great game!", 
            "raw_history": [m.dict() for m in request.history]
        }

    history_summary = "\n".join([f"Step {m.step}: {m.evaluation_label} - {m.comment}" for m in request.history])
    report_prompt = f"Provide a brief Executive Summary (max 100 words) of this match. Result: {request.final_result}. History: {history_summary[-1000:]}"
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=report_prompt)
        return {"report_text": response.text, "raw_history": [m.dict() for m in request.history]}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate report.")

# --- AI SCHEDULED BATTLE ENDPOINTS ---

@app.post("/api/schedule-match")
async def schedule_ai_match(request: MatchScheduleRequest, db: Session = Depends(get_db), current_user: UserORM = Depends(get_current_active_user)):
    new_match = ScheduledMatchORM(user_id=current_user.id, scheduled_time=request.scheduled_time, status="PENDING")
    db.add(new_match)
    db.commit()
    logger.info(f"User {current_user.username} scheduled a match for {request.scheduled_time}")
    return {"message": "Match scheduled successfully", "id": new_match.id}

@app.get("/api/notifications")
async def get_match_notifications(db: Session = Depends(get_db), current_user: UserORM = Depends(get_current_active_user)):
    completed_matches = db.query(ScheduledMatchORM).filter(ScheduledMatchORM.user_id == current_user.id, ScheduledMatchORM.status == "COMPLETED", ScheduledMatchORM.is_notified == False).all()
    notifications = [{"id": m.id, "result": m.final_result, "time": m.scheduled_time} for m in completed_matches]
    for m in completed_matches: m.is_notified = True
    db.commit()
    return {"notifications": notifications}

@app.get("/api/scheduled-report/{match_id}")
async def get_scheduled_report(match_id: int, db: Session = Depends(get_db), current_user: UserORM = Depends(get_current_active_user)):
    match = db.query(ScheduledMatchORM).filter(ScheduledMatchORM.id == match_id, ScheduledMatchORM.user_id == current_user.id).first()
    if not match or not match.match_data: raise HTTPException(status_code=404, detail="Match report not found")
    return {"history": match.match_data, "final_result": match.final_result}

@app.post("/api/internal/execute-matches")
async def execute_scheduled_matches(x_cron_secret: str = Header(None), db: Session = Depends(get_db)):
    if x_cron_secret != CRON_SECRET: raise HTTPException(status_code=403, detail="Unauthorized internal call")
    now = datetime.utcnow()
    pending_matches = db.query(ScheduledMatchORM).filter(ScheduledMatchORM.status == "PENDING", ScheduledMatchORM.scheduled_time <= now).all()
    executed_count = 0
    for match in pending_matches:
        try:
            simulator = GomokuSimulator()
            history, result = simulator.run_match()
            match.match_data = history
            match.final_result = result
            match.status = "COMPLETED"
            executed_count += 1
        except Exception as e:
            logger.error(f"Failed to execute match {match.id}: {e}")
            match.status = "FAILED"
    db.commit()
    return {"executed_matches": executed_count}

@app.post("/api/internal/cleanup-data")
async def cleanup_old_data(x_cron_secret: str = Header(None), db: Session = Depends(get_db)):
    if x_cron_secret != CRON_SECRET: raise HTTPException(status_code=403, detail="Unauthorized internal call")
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    try:
        deleted_count = db.query(ScheduledMatchORM).filter(ScheduledMatchORM.created_at < cutoff_date).delete()
        db.commit()
        return {"status": "success", "deleted_records": deleted_count, "cutoff": cutoff_date.isoformat()}
    except Exception as e:
        logger.error(f"Data cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Cleanup failed")