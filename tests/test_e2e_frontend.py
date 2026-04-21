# tests/test_e2e_frontend.py
# ---------------------------------------------------------
# Project: Gomoku AI Coach (15x15)
# Feature: Frontend E2E Tests (Playwright)
# ---------------------------------------------------------

import pytest
import multiprocessing
import uvicorn
import time
from playwright.sync_api import Page, expect

# --- 本地测试服务器管理 ---
def run_server():
    """在后台运行真实的 FastAPI 服务器供浏览器访问"""
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