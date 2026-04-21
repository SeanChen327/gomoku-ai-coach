# tests/test_e2e_frontend.py
# ---------------------------------------------------------
# Project: Gomoku AI Coach (15x15)
# Feature: Frontend E2E Tests (Playwright)
# ---------------------------------------------------------

import pytest
import multiprocessing
import uvicorn
import time
import os
from playwright.sync_api import Page, expect

# --- 本地测试服务器管理 ---
def run_server():
    """在后台运行真实的 FastAPI 服务器供浏览器访问"""
    # [QA Gate] 强制注入 MOCK_AI 环境变量给子进程，避免 E2E 测试耗尽真实 API 配额
    os.environ["MOCK_AI"] = "true"
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="critical")

@pytest.fixture(scope="module", autouse=True)
def live_server():
    """
    Pytest Fixture: 在执行所有 E2E 测试前，自动启动 FastAPI 服务器。
    测试结束后自动关闭。
    """
    proc = multiprocessing.Process(target=run_server, daemon=True)
    proc.start()
    time.sleep(2)  # 等待服务器完全启动
    yield "http://127.0.0.1:8000"
    proc.terminate()

# --- E2E 测试用例 ---
class TestFrontendE2E:
    """
    端到端测试套件：模拟真实用户在浏览器中的交互。
    """

    def test_initial_load_unauthenticated(self, page: Page, live_server: str):
        """
        验证点：未登录用户访问页面时，游戏区域应被隐藏，并提示登录。
        """
        page.goto(live_server)

        # 1. 验证标题加载正确
        expect(page).to_have_title("Gomoku AI Coach (15x15)")

        # 2. 验证 Auth 状态显示为未登录
        status_display = page.locator("#auth-status-display")
        expect(status_display).to_contain_text("Not Logged In")

        # 3. 验证游戏棋盘不可见 (被 CSS 隐藏)
        game_area = page.locator("#game-area")
        expect(game_area).not_to_be_visible()

        # 4. 验证登录提示文案可见
        login_prompt = page.locator("#login-prompt")
        expect(login_prompt).to_be_visible()
        expect(login_prompt).to_contain_text("Welcome!")

    def test_auth_modal_interaction(self, page: Page, live_server: str):
        """
        验证点：点击 Login 按钮后，能够正确弹出鉴权模态框，并能切换注册/登录视图。
        """
        page.goto(live_server)

        # 1. 点击顶部的 Login 按钮
        page.locator("#auth-action-btn").click()

        # 2. 验证 Modal 弹出并显示 Login 表单
        modal = page.locator("#auth-modal")
        expect(modal).to_be_visible()
        expect(page.locator("#auth-title")).to_have_text("🔐 Login")
        expect(page.locator("#login-form")).to_be_visible()

        # 3. 点击切换到注册视图
        page.locator("#login-form .auth-toggle-link").click()
        
        # 4. 验证 UI 切换为 Sign Up 表单
        expect(page.locator("#auth-title")).to_have_text("📝 Sign Up")
        expect(page.locator("#register-form")).to_be_visible()
        expect(page.locator("#login-form")).not_to_be_visible()

        # 5. 测试关闭功能
        page.locator("#auth-modal .close-btn").click()
        expect(modal).not_to_be_visible()
    
    def test_core_gameplay_and_ai_chat(self, page: Page, live_server: str):
        """
        验证点：真实用户视角下的核心业务流。
        涵盖：UI注册登录 -> 玩家落子 -> AI启发式回棋 -> 调出AI教练追问对局状况。
        """
        page.goto(live_server)

        # ---------------------------------------------------------
        # 1. 极速 UI 注册与登录 (使用时间戳防止用户名冲突)
        # ---------------------------------------------------------
        username = f"player_{int(time.time())}"
        
        # 打开弹窗 -> 切换到注册
        page.locator("#auth-action-btn").click()
        page.locator("#login-form .auth-toggle-link").click()
        
        # 填写注册信息并提交
        page.locator("#register-username").fill(username)
        page.locator("#register-email").fill(f"{username}@example.com")
        page.locator("#register-password").fill("E2eTestPass123!")
        page.locator("#register-form .btn-success").click()

        # 等待自动切回登录页面，填入信息并登录
        page.locator("#login-username").fill(username)
        page.locator("#login-password").fill("E2eTestPass123!")
        page.locator("#login-form .btn-primary").click()

        # 验证：登录成功，游戏棋盘成功渲染
        game_area = page.locator("#game-area")
        expect(game_area).to_be_visible(timeout=5000)

        # ---------------------------------------------------------
        # 2. 模拟真实下棋交互 (人机大战)
        # ---------------------------------------------------------
        # 点击天元位置 (中心点 H8，索引 112)
        center_cell = page.locator("#cell-112")
        center_cell.click()

        # 验证：玩家的黑子 (X) 已正确渲染到棋盘上
        expect(center_cell.locator(".stone-X")).to_be_visible()

        # 验证：系统调用了 getHeuristicMove()，并且 AI 最终落下了白子 (O)
        # (只要页面上出现了任何一个 .stone-O，就说明前端逻辑跑通了)
        expect(page.locator(".stone-O").first).to_be_visible(timeout=3000)

        # ---------------------------------------------------------
        # 3. 验证 AI 教练 (LangChain + Gemini RAG) 通讯
        # ---------------------------------------------------------
        # 唤起聊天气泡
        page.locator("#chat-toggle").click()
        expect(page.locator("#chat-container")).to_be_visible()

        # 输入测试问题
        chat_input = page.locator("#chat-input")
        chat_input.fill("What is my win rate right now?")
        
        # 点击发送
        page.locator("#chat-input-area button").click()

        # 验证：用户的消息成功上屏
        expect(page.locator("#chat-history")).to_contain_text("What is my win rate right now?")

        # 验证：AI 的最终回复成功上屏 (避开 "Analyzing board..." 的 Loading 状态)
        # 开启了 MOCK_AI 挡板，这里返回 mock 数据。给足 15 秒 Timeout。
        ai_responses = page.locator("#chat-history .ai-msg")
        expect(ai_responses.last).not_to_contain_text("Analyzing board...", timeout=15000)
        # 确保它确实回了一段话 (长度大于 0)
        expect(ai_responses.last).to_be_visible()