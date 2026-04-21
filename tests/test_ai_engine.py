# tests/test_ai_engine.py
# ---------------------------------------------------------
# Project: Gomoku AI Coach (15x15)
# Feature: Core Engine Unit Tests (Mathematics & Logic)
# ---------------------------------------------------------

import pytest
from ai_battle_engine import GomokuSimulator, BOARD_SIZE, TOTAL_CELLS

class TestGomokuSimulator:
    """
    Unit test suite for the offline Gomoku simulator engine.
    Ensures mathematical accuracy and deterministic heuristic evaluation.
    """

    @pytest.fixture
    def simulator(self):
        """Returns a fresh instance of the simulator before each test."""
        return GomokuSimulator()

    def test_index_to_coord(self, simulator):
        """Verifies 1D index to 2D coordinate translation."""
        assert simulator.index_to_coord(0) == "A1", "Top-left corner failed"
        assert simulator.index_to_coord(14) == "O1", "Top-right corner failed"
        assert simulator.index_to_coord(210) == "A15", "Bottom-left corner failed"
        assert simulator.index_to_coord(224) == "O15", "Bottom-right corner failed"

    def test_check_winner_horizontal(self, simulator):
        """Verifies horizontal win detection."""
        # Setup: X has 5 in a row horizontally starting at index 0 (A1 to E1)
        for i in range(5):
            simulator.board[i] = "X"
        
        assert simulator.check_winner(simulator.board) == "X", "Failed to detect horizontal win for X"

    def test_check_winner_diagonal(self, simulator):
        """Verifies diagonal win detection."""
        # Setup: O has 5 in a row diagonally (A1, B2, C3, D4, E5)
        for i in range(5):
            idx = i * BOARD_SIZE + i
            simulator.board[idx] = "O"
            
        assert simulator.check_winner(simulator.board) == "O", "Failed to detect diagonal win for O"

    def test_heuristic_critical_defense(self, simulator):
        """
        Verifies the engine correctly assigns a massive score (>=10000) 
        to block an opponent's 4-in-a-row (Lethal Threat).
        """
        # Setup: X has 4 in a row (B2, C2, D2, E2). 
        # A2 (index 15) and F2 (index 20) are open.
        simulator.board[16] = "X" # B2
        simulator.board[17] = "X" # C2
        simulator.board[18] = "X" # D2
        simulator.board[19] = "X" # E2

        # Evaluate the empty spot at A2 (index 15) from O's perspective (evaluating X's threat)
        threat_score = simulator.evaluate_cell(simulator.board, 15, "X")
        
        # An open 4-in-a-row should yield at least 10000 points
        assert threat_score >= 10000, f"Critical threat undervalued! Score was only {threat_score}"

    def test_heuristic_open_three(self, simulator):
        """
        Verifies the engine correctly scores an 'Open Three' pattern (~500 points).
        """
        # Setup: O has 3 in a row (C3, D3, E3).
        # B3 (index 31) and F3 (index 35) are open.
        simulator.board[32] = "O" # C3
        simulator.board[33] = "O" # D3
        simulator.board[34] = "O" # E3

        # Evaluate the empty spot at B3 (index 31) for player O
        offensive_score = simulator.evaluate_cell(simulator.board, 31, "O")
        
        # An open 3-in-a-row should yield roughly 500 points
        assert offensive_score >= 500, f"Open three undervalued! Score was {offensive_score}"

    def test_get_best_move_blocks_win(self, simulator):
        """
        E2E Logic Test: Ensure the AI will prioritize blocking a win over a random move.
        """
        # Setup: X is about to win horizontally at the top.
        simulator.board[0] = "X"
        simulator.board[1] = "X"
        simulator.board[2] = "X"
        simulator.board[3] = "X"
        # The blocking move is index 4 (E1).

        best_move = simulator.get_best_move(simulator.board, "O")
        assert best_move == 4, f"AI failed to block the winning move! It chose {best_move} instead."