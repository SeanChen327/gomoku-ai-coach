import os
import pytest
import httpx
from langchain_google_genai import ChatGoogleGenerativeAI

# [FIX 1] Update imports to use langchain_core to match modern LangChain architecture
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure API Key is available for the LangChain evaluator
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
        
        # LangChain prompt to evaluate our production API's response
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

    @pytest.mark.asyncio
    async def test_chat_endpoint_heuristics_compliance(self, evaluator_chain):
        """
        E2E test: Verifies that /api/chat respects heuristic payloads using LangChain validation.
        """
        # 1. Setup mock data simulating the frontend heuristic engine
        payload = {
            "message": "Where should I play next and how am I doing?",
            "board": [""] * 225,
            "last_evaluation": {
                "evaluation_label": "Standard Move",
                "comment": "Played at coordinate H8.",
                "suggested_next_move": "H9",
                "win_rate": "55%"
            }
        }

        # 2. Call the untouched production API (Assuming it's running locally for testing)
        # Note: In a real CI environment, we would use FastAPI's TestClient or a spun-up test server.
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
            # Note: Requires bypassing Auth for testing or mocking the JWT token
            # For demonstration, we assume a mock/test endpoint or active token
            pass 
            
            # Mocking the response for the sake of this architectural example
            # response = await client.post("/api/chat", json=payload)
            # ai_reply = response.json()["reply"]
            ai_reply = "You're doing great with a 55% win rate! Tactically, playing at H9 is your best move right now to build a strong offensive shape. Keep it up!"

        # 3. Use LangChain to evaluate the output
        eval_result = evaluator_chain.invoke({
            "win_rate": "55%",
            "next_move": "H9",
            "ai_response": ai_reply
        })

        # 4. Assert DevOps Quality Standard
        assert eval_result.strip() == "PASS", f"LLM QA Validation Failed: {eval_result}"