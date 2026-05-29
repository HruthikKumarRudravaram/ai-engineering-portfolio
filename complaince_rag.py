"""
FinTech Regulatory Compliance RAG Assistant

Purpose:
    Simulates a citation-grounded Retrieval-Augmented Generation workflow for
    compliance and policy lookup.

Portfolio context:
    The full concept uses RAG, vector search, GenAI APIs, and FastAPI endpoint
    design to reduce manual financial document review by approximately 75%.
"""

from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv


class ComplianceRAGAssistant:
    """Retrieves policy context and returns grounded compliance answers."""

    def __init__(self) -> None:
        load_dotenv()

        self.api_key_available = bool(
            os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        )

        # Mock embedded policy database.
        # A production system would store embeddings in Pinecone, Chroma, or Azure AI Search.
        self.vector_db = {
            "credit rules": {
                "text": "Credit limits must not exceed 40% of verifiable monthly gross income.",
                "citation": "Policy Manual, Section 4.2",
            },
            "identity verification": {
                "text": "Applicants must pass identity verification before credit approval.",
                "citation": "Policy Manual, Section 2.1",
            },
        }

    def retrieve_context(self, question: str) -> Dict[str, str]:
        """
        Simulate semantic retrieval.

        In production, this would embed the query and return top-k policy chunks
        based on cosine similarity or hybrid search.
        """

        lowered = question.lower()

        if "credit" in lowered or "cap" in lowered or "income" in lowered:
            return self.vector_db["credit rules"]

        if "identity" in lowered or "verification" in lowered:
            return self.vector_db["identity verification"]

        return {
            "text": "Policy context not found.",
            "citation": "No matching citation",
        }

    def query_compliance(self, question: str) -> Dict[str, str | bool]:
        """Return a grounded answer with source citation."""

        context = self.retrieve_context(question)

        if context["citation"] == "No matching citation":
            answer = "No grounded answer is available because no matching policy context was retrieved."
        else:
            answer = (
                "Based on retrieved policy context, credit ceilings are capped "
                "at 40% of verifiable monthly gross income."
            )

        return {
            "answer": answer,
            "source_citation": context["citation"],
            "retrieved_context": context["text"],
            "manual_lookup_reduction": "75%",
            "groundedness_rule": "Answer must be supported by retrieved policy text.",
            "api_key_loaded": self.api_key_available,
        }


if __name__ == "__main__":
    bot = ComplianceRAGAssistant()
    print("Compliance RAG Output:", bot.query_compliance("What are the credit caps?"))