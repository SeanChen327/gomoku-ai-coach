# main.py
import logging
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from google import genai 
from dotenv import load_dotenv
from passlib.context import CryptContext
from jose import JWTError, jwt
from ai_governance import GomokuAIGovernance # [NEW] Import the governance module
import time # Ensure time is imported for latency tracking

# --- SQLAlchemy Imports ---
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- [NEW] 引入离线对弈引擎 ---
from ai_battle_engine import GomokuSimulator

# [SECURITY UPDATE] Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# [DevOps] 全局控制：是否启用 AI 挡板模式 (Mock Mode)
# 当在 CI/CD 测试环境中注入 MOCK_AI="true" 时，将跳过真实的大模型调用以保护 API Quota
MOCK_AI = os.getenv("MOCK_AI", "false").lower() == "true"

# --- DATABASE CONFIGURATION ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tictactoe.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DB MODELS ---
class UserORM(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

class ScheduledMatchORM(Base):
    """
    [NEW] 存储 AI 预约对局数据。
    """
    __tablename__ = "scheduled_matches"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="PENDING")  # PENDING, COMPLETED, FAILED
    scheduled_time = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    
    # 存储对局历史 (JSON 格式，复用前端 gameHistory 结构)
    match_data = Column(JSON, nullable=True) 
    final_result = Column(String, nullable=True)
    
    # 消息提醒状态
    is_notified = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Instantiate the governance layer based on the AI Periodic Table
governance = GomokuAIGovernance()

# [NEW] 从环境变量读取 Cron 密钥
CRON_SECRET = os.getenv("CRON_SECRET", "default_internal_secret_change_me")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    # 如果不是 Mock 模式，才强制抛出异常；方便 CI 环境无密钥也能跑通部分基础测试
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
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str
    board: List[str]
    last_evaluation: Optional[Dict[str, Any]] = None

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

# [NEW] 预约对局请求模型
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
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
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

# --- HEURISTIC ANALYSIS ---
def analyze_board(board: List[str]) -> str:
    BOARD_SIZE = 15
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            idx = r * BOARD_SIZE + c
            player = board[idx]
            if not player: continue
            for dr, dc in directions:
                count = 1
                for step in range(1, 4):
                    nr, nc = r + dr * step, c + dc * step
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr * BOARD_SIZE + nc] == player: count += 1
                        else: break
                    else: break
                if count == 4:
                    if player == "O": return f"[CRITICAL]: Computer (O) has 4 in a row near index {idx}! Block immediately."
                    if player == "X": return f"[OPPORTUNITY]: Human (X) has 4 in a row near index {idx}! Push for the win."
    return "[NEUTRAL]: No immediate 4-in-a-row threats detected. Focus on building open threes."

# --- LANGCHAIN RAG SERVICE ---
class GomokuRagService:
    def __init__(self):
        # 如果是 MOCK 模式且没有 API KEY，则跳过初始化，防止报错
        if MOCK_AI and not GEMINI_API_KEY:
            return
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=GEMINI_API_KEY
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            output_dimensionality=768
        )
        self.vectorstore = PineconeVectorStore(
            index_name="tictactoe-rag",
            embedding=self.embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

    def get_chain(self):
        prompt = ChatPromptTemplate.from_template("""
        [ROLE]: You are a professional 15x15 Gomoku (Five-in-a-Row) AI Coach. 
        Human plays Black stones ('X'), Computer plays White stones ('O').
        
        [TACTICAL ANALYSIS]: {tactical_analysis}
        [RECENT MOVE CONTEXT]: {move_context}
        [RAG EXPERT CONTEXT]: {rag_context}
        
        User Question: "{message}"
        
        [STRICT GUIDELINES]:
        1. Acknowledge this is Gomoku (15x15 grid).
        2. Only rely on the provided metrics. Do not hallucinate coordinates.
        3. If asked about the LAST MOVE: Evaluate it based on the recent move comment.
        4. If asked about the NEXT MOVE: Use the suggested coordinate from context.
        5. Keep response highly encouraging and under 80 words.
        """)

        def format_move_context(input_data):
            le = input_data.get('last_evaluation')
            if not le: return "No recent move data available."
            ctx = f"Player X's last move (Index {le.get('index', 'N/A')}) was: [{le.get('evaluation_label')}]. Comment: '{le.get('comment')}'. "
            if le.get('suggested_next_move'):
                ctx += f"Suggested next tactical move: {le.get('suggested_next_move')}. "
            if le.get('win_rate'):
                ctx += f"Current win rate: {le.get('win_rate')}. "
            return ctx

        chain = (
            {
                "rag_context": (lambda x: x["message"]) | self.retriever | (lambda docs: "\n".join([d.page_content for d in docs])),
                "tactical_analysis": RunnableLambda(lambda x: analyze_board(x["board"])),
                "move_context": RunnableLambda(format_move_context),
                "message": lambda x: x["message"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

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

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest, current_user: UserORM = Depends(get_current_active_user)):
    logger.info(f"User '{current_user.username}' requested AI chat.")
    start_t = time.time()
    
    # 👇 [QA Gate] If running in CI/CD mock mode
    if MOCK_AI:
        await asyncio.sleep(1) 
        reply_text = "You're doing great with a 55% win rate! Tactically, playing at H9 is your best move right now to build a strong offensive shape. Keep it up!"
    else:
        try:
            reply_data = await rag_service.get_chain().ainvoke({
                "message": request.message,
                "board": request.board,
                "last_evaluation": request.last_evaluation
            })
            reply_text = reply_data
        except Exception as e:
            logger.error(f"Chat Error: {str(e)}")
            return {"reply": "The AI coach is temporarily unavailable.", "telemetry": {}}

    # ---------------------------------------------------------
    # 🛡️ Validation Layer (Guardrails) 
    # ---------------------------------------------------------
    if not governance.validate_output_safety(reply_text, request.board):
        # Fallback: Do not return erroneous coordinate hallucinations to the user
        reply_text = "I apologize, my coordinate analysis was slightly off just now. Could you rephrase your question?"

    # ---------------------------------------------------------
    # 📊 Assurance Layer (Metrics + Evaluation)
    # ---------------------------------------------------------
    metrics = governance.track_telemetry(start_t, reply_text)
    quality_score = governance.evaluate_response_consistency(reply_text, request.last_evaluation)

    # Record production-grade logs including telemetry and quality scoring
    logger.info(
        f"[PROD_LOG] User: {current_user.username} | "
        f"Quality Score: {quality_score:.2f} | "
        f"Metrics: {metrics}"
    )

    return {"reply": reply_text, "telemetry": metrics}

@app.post("/api/generate-report")
async def generate_report(request: GameReportRequest, current_user: UserORM = Depends(get_current_active_user)):
    # 👇 [QA Gate] 同样为生成报告接口提供 Mock 保护
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

# --- [NEW] AI SCHEDULED BATTLE ENDPOINTS ---

@app.post("/api/schedule-match")
async def schedule_ai_match(
    request: MatchScheduleRequest, 
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_active_user)
):
    """
    用户预约一场未来的 AI 对决。
    """
    new_match = ScheduledMatchORM(
        user_id=current_user.id,
        scheduled_time=request.scheduled_time,
        status="PENDING"
    )
    db.add(new_match)
    db.commit()
    logger.info(f"User {current_user.username} scheduled a match for {request.scheduled_time}")
    return {"message": "Match scheduled successfully", "id": new_match.id}

@app.get("/api/notifications")
async def get_match_notifications(
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_active_user)
):
    """
    检查是否有已完成且未通知的对局。
    """
    completed_matches = db.query(ScheduledMatchORM).filter(
        ScheduledMatchORM.user_id == current_user.id,
        ScheduledMatchORM.status == "COMPLETED",
        ScheduledMatchORM.is_notified == False
    ).all()
    
    notifications = []
    for m in completed_matches:
        notifications.append({
            "id": m.id,
            "result": m.final_result,
            "time": m.scheduled_time
        })
        # 标记为已通知
        m.is_notified = True
    
    db.commit()
    return {"notifications": notifications}

@app.get("/api/scheduled-report/{match_id}")
async def get_scheduled_report(
    match_id: int,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_active_user)
):
    """
    获取预约对弈的结果历史。
    """
    match = db.query(ScheduledMatchORM).filter(
        ScheduledMatchORM.id == match_id,
        ScheduledMatchORM.user_id == current_user.id
    ).first()
    
    if not match or not match.match_data:
        raise HTTPException(status_code=404, detail="Match report not found")
    
    return {"history": match.match_data, "final_result": match.final_result}

@app.post("/api/internal/execute-matches")
async def execute_scheduled_matches(
    x_cron_secret: str = Header(None), 
    db: Session = Depends(get_db)
):
    """
    [INTERNAL] 查找并执行所有到期的预约。
    """
    if x_cron_secret != CRON_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized internal call")

    now = datetime.utcnow()
    pending_matches = db.query(ScheduledMatchORM).filter(
        ScheduledMatchORM.status == "PENDING",
        ScheduledMatchORM.scheduled_time <= now
    ).all()

    executed_count = 0
    for match in pending_matches:
        try:
            # 运行模拟器
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

# 👇 [核心新增] 数据生命周期管理接口 (Data Retention Cleanup)
@app.post("/api/internal/cleanup-data")
async def cleanup_old_data(
    x_cron_secret: str = Header(None), 
    db: Session = Depends(get_db)
):
    """
    [INTERNAL] 删除 30 天前的陈旧对局数据，防止数据库存储溢出。
    由 GitHub Actions 定时任务触发。
    """
    if x_cron_secret != CRON_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized internal call")

    # 设定清理阈值：30天前
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    
    try:
        deleted_count = db.query(ScheduledMatchORM).filter(
            ScheduledMatchORM.created_at < cutoff_date
        ).delete()
        db.commit()
        logger.info(f"Data cleanup executed. Removed {deleted_count} old matches.")
        return {"status": "success", "deleted_records": deleted_count, "cutoff": cutoff_date.isoformat()}
    except Exception as e:
        logger.error(f"Data cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Cleanup failed")