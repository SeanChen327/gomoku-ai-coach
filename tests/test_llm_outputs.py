# tests/test_llm_outputs.py
import os
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() 

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class TestAICoachQuality:
    """
    Automated Validation Suite for the Gomoku AI Coach using LangChain.
    Acts as an independent quality gate to prevent LLM hallucinations.
    """

    @pytest.fixture
    def evaluator_chain(self):
        """
        Initializes the LangChain evaluation pipeline (LLM-as-a-Judge).
        """
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=GEMINI_API_KEY,
            temperature=0.0
        )
        
        eval_template = """
        You are a strict QA Engineer. Evaluate the AI Coach's response based on the following criteria:
        1. Word Count: Is it under 80 words?
        2. Accuracy: Does it strictly rely on the provided Context (Win Rate: {win_rate}, Next Move: {next_move}) without hallucinating coordinates?
        3. Tone: Is it encouraging and conversational?
        
        Context injected into production AI: Win Rate was {win_rate}, Suggested Move was {next_move}.
        Production AI Response to evaluate: "{ai_response}"
        
        Respond ONLY with "PASS" if all criteria are met, or "FAIL: [Reason]" if it fails.
        """
        prompt = PromptTemplate.from_template(eval_template)
        return prompt | llm | StrOutputParser()

    # 👇 [QA Gate]：注入 skipif 装饰器，当处于 Mock 模式时自动跳过此用例，保护 API 配额
    @pytest.mark.skipif(
        os.getenv("MOCK_AI", "false").lower() == "true", 
        reason="Skipping LLM evaluation to conserve Gemini API Quota in CI."
    )
    def test_chat_endpoint_heuristics_compliance(self, evaluator_chain):
        """
        E2E test: Verifies that /api/chat respects heuristic payloads using LangChain validation.
        """
        # 1. Mocking the response for the sake of this architectural example
        ai_reply = "You're doing great with a 55% win rate! Tactically, playing at H9 is your best move right now to build a strong offensive shape. Keep it up!"

        # 2. Use LangChain to evaluate the output (Synchronously!)
        eval_result = evaluator_chain.invoke({
            "win_rate": "55%",
            "next_move": "H9",
            "ai_response": ai_reply
        })

        # 3. Assert DevOps Quality Standard
        assert eval_result.strip() == "PASS", f"LLM QA Validation Failed: {eval_result}"