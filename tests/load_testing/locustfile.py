# tests/load_testing/locustfile.py
"""
Project: Gomoku AI Coach (15x15)
Feature: Load Testing Suite
Usage: 
  1. pip install locust
  2. export MOCK_AI=true && uvicorn main:app --port 8000
  3. locust -f tests/load_testing/locustfile.py --host http://127.0.0.1:8000
"""

from locust import HttpUser, task, between
import random
import string
from datetime import datetime, timedelta

def random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

class GomokuLoadUser(HttpUser):
    # 模拟真实用户的思考和操作延迟（1-3秒）
    wait_time = between(1.0, 3.0)
    
    def on_start(self):
        """
        Setup: Register and Login a temporary user for each simulated client.
        """
        self.username = f"loaduser_{random_string()}"
        self.password = "LoadTest123!"
        
        # 1. 注册
        self.client.post("/api/register", json={
            "username": self.username,
            "email": f"{self.username}@loadtest.com",
            "password": self.password
        })
        
        # 2. 登录拿 Token
        response = self.client.post("/api/token", data={
            "username": self.username,
            "password": self.password
        })
        
        if response.status_code == 200:
            token = response.json().get("access_token")
            self.headers = {"Authorization": f"Bearer {token}"}
        else:
            self.headers = {}

    @task(3)
    def check_health(self):
        """模拟高频健康检查 / 首页加载"""
        self.client.get("/api/health")

    @task(2)
    def chat_with_ai(self):
        """
        模拟用户向 AI 提问 (由于环境变量 MOCK_AI=true，此请求将秒回且不耗费 API Quota)
        """
        payload = {
            "message": "What is my win rate?",
            "board": [""] * 225,
            "last_evaluation": None
        }
        self.client.post("/api/chat", json=payload, headers=self.headers)

    @task(1)
    def schedule_battle(self):
        """模拟低频的定时对局预约"""
        future_time = (datetime.utcnow() + timedelta(days=1)).isoformat()
        self.client.post("/api/schedule-match", json={"scheduled_time": future_time}, headers=self.headers)