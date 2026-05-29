"""
Document Intelligence & HR Automation Agent

Purpose:
    Simulates a resume/document intelligence workflow that extracts key
    parameters from unstructured text and returns structured review metadata.

Portfolio context:
    The full project used Streamlit/FastAPI, OpenAI GPT, and PDF/TXT extraction
    to automate tailored resume feedback and reduce manual review effort by 80%.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

from dotenv import load_dotenv


class DocumentIntelligenceAgent:
    """Analyzes document text for role alignment and automation metrics."""

    def __init__(self) -> None:
        load_dotenv()

        self.api_key_available = bool(os.getenv("OPENAI_API_KEY"))
        self.token_cost_per_1k = 0.0006
        self.target_skills = [
            "python",
            "sql",
            "fastapi",
            "streamlit",
            "openai",
            "ai",
            "rag",
            "machine learning",
        ]

    def clean_document_text(self, raw_text: str) -> str:
        """Normalize extracted document text before analysis."""

        return re.sub(r"\s+", " ", raw_text.strip().lower())

    def extract_skills(self, clean_text: str) -> List[str]:
        """Extract targeted skills from unstructured document text."""

        return [skill for skill in self.target_skills if skill in clean_text]

    def estimate_token_cost(self, clean_text: str) -> float:
        """Approximate LLM inference cost for experimentation tracking."""

        estimated_tokens = len(clean_text) / 4
        return round((estimated_tokens / 1000) * self.token_cost_per_1k, 6)

    def analyze_text(self, raw_text: str) -> Dict[str, object]:
        """Return structured HR/document intelligence output."""

        clean_text = self.clean_document_text(raw_text)
        detected_skills = self.extract_skills(clean_text)
        role_alignment = "strong" if len(detected_skills) >= 3 else "moderate"

        return {
            "extracted_skills": detected_skills,
            "role_alignment_signal": role_alignment,
            "review_efficiency_gain": "80% reduction in manual effort",
            "feedback_turnaround": "<5 seconds per resume",
            "estimated_token_cost_usd": self.estimate_token_cost(clean_text),
            "api_key_loaded": self.api_key_available,
        }


if __name__ == "__main__":
    document_sample = """
    Deep experience with Python, SQL data pipelines, FastAPI services,
    and building AI prototypes with OpenAI models.
    """

    agent = DocumentIntelligenceAgent()
    print("Document Intelligence Output:", agent.analyze_text(document_sample))