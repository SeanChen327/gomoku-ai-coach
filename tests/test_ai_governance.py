# tests/test_ai_governance.py
import pytest
import time
from ai_governance import GomokuAIGovernance

class TestAIGovernance:
    @pytest.fixture
    def governance(self):
        return GomokuAIGovernance()

    @pytest.fixture
    def empty_board(self):
        # 15x15 五子棋盘，共 225 个空位
        return [""] * 225

    def test_guardrails_safe_move(self, governance, empty_board):
        """测试正常坐标是否通过验证"""
        safe_reply = "I suggest playing at H8."
        assert governance.validate_output_safety(safe_reply, empty_board) is True

    def test_guardrails_out_of_bounds(self, governance, empty_board):
        """测试越界坐标幻觉拦截 (例如 Z99)"""
        hallucination_reply = "Try placing your stone at Z99."
        assert governance.validate_output_safety(hallucination_reply, empty_board) is False

    def test_guardrails_occupied_position(self, governance, empty_board):
        """测试占用位置幻觉拦截"""
        # 模拟 H8 (索引 112) 已有棋子
        empty_board[112] = "X"
        occupied_reply = "You should definitely play at H8."
        assert governance.validate_output_safety(occupied_reply, empty_board) is False

    def test_metrics_tracking(self, governance):
        """测试遥测数据计算是否正确"""
        start_time = time.time() - 0.5 # 模拟 0.5秒的延迟
        reply = "Great move!"
        
        metrics = governance.track_telemetry(start_time, reply)
        
        assert "latency_ms" in metrics
        assert metrics["latency_ms"] >= 500
        assert metrics["estimated_tokens"] == len(reply) * 2
        assert "estimated_cost" in metrics

    def test_evaluation_consistency(self, governance):
        """测试 AI 回复的胜率与启发式数据的对齐度"""
        last_eval = {"win_rate": "60%"}
        
        # 1. 完全一致
        score_perfect = governance.evaluate_response_consistency("Your win rate is 60%.", last_eval)
        assert score_perfect == 1.0
        
        # 2. 存在偏差 (AI 幻觉说 80%)
        score_deviated = governance.evaluate_response_consistency("You have an 80% chance.", last_eval)
        assert score_deviated == 0.8  # 1.0 - (20/100)