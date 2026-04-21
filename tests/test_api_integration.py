# tests/test_api_integration.py
# ---------------------------------------------------------
# Project: Gomoku AI Coach (15x15)
# Feature: Backend API Integration Tests (Fixed Fixture Dependencies)
# ---------------------------------------------------------

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

# 导入生产环境的代码
from main import app, get_db, Base

# --- 测试环境配置 ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_integration.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client(test_db):
    return TestClient(app)

@pytest.fixture(scope="module")
def setup_base_user(client):
    """
    独立的基础用户创建 Fixture。
    保证在整个模块中只注册一次，为其他测试提供确定的基准状态。
    """
    user_data = {
        "username": "qa_automation_user",
        "email": "qa@example.com",
        "password": "SuperSecretPassword123!"
    }
    client.post("/api/register", json=user_data)
    return user_data

@pytest.fixture(scope="module")
def test_user_token(client, setup_base_user):
    """
    依赖于 setup_base_user 的 Token Fixture。
    确切知道密码是什么，保证必定登录成功。
    """
    response = client.post(
        "/api/token", 
        data={"username": setup_base_user["username"], "password": setup_base_user["password"]}
    )
    return response.json()["access_token"]


class TestAPIIntegration:
    
    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200, "Health check failed"
        assert response.json()["status"] == "online", "Server status is not online"

    # [修复点] 这里显式引入 setup_base_user，强制 Pytest 在运行此测试前先创建用户
    def test_register_duplicate_user(self, client, setup_base_user):
        """Validates that registering a user with an existing username throws a 400 error."""
        user_data = {
            "username": setup_base_user["username"], 
            "email": "another_qa@example.com",
            "password": "Password123"
        }
        response = client.post("/api/register", json=user_data)
        assert response.status_code == 400, "API did not block duplicate registration"
        assert "already registered" in response.json()["detail"].lower()

    def test_schedule_match_unauthorized(self, client):
        """Security Validation: Ensure match scheduling is blocked without a valid JWT."""
        future_time = (datetime.utcnow() + timedelta(days=1)).isoformat()
        response = client.post("/api/schedule-match", json={"scheduled_time": future_time})
        assert response.status_code == 401, "API allowed unauthorized access to schedule-match"

    def test_schedule_match_success(self, client, test_user_token):
        """Validates that an authenticated user can successfully schedule an AI match."""
        headers = {"Authorization": f"Bearer {test_user_token}"}
        future_time = (datetime.utcnow() + timedelta(days=1)).isoformat()
        
        response = client.post(
            "/api/schedule-match", 
            json={"scheduled_time": future_time}, 
            headers=headers
        )
        assert response.status_code == 200, f"Match scheduling failed: {response.text}"
        assert response.json()["message"] == "Match scheduled successfully"
        assert "id" in response.json(), "Match ID was not returned by the API"