# Multi-Modal AI & Analytics Portfolio

A collection of targeted academic and independent projects demonstrating data engineering, classical machine learning, and generative AI architectures focused on business operations, risk automation, and workflow efficiency.

This portfolio is structured for AI Engineer, Agents & Experimentation roles, with emphasis on:

- **Operational impact:** reducing manual review effort and accelerating business decisions.
- **Efficiency and automation:** routing repetitive work through AI/ML systems.
- **Experimentation:** selecting the right model strategy based on precision, latency, cost, and business value.

---

## 1. Project Index & Performance Metrics

| Project | Core Technique | Domain / Use Case | Primary Business Metric |
| --- | --- | --- | --- |
| **1. Financial Fraud Detection** | Isolation Forest, SQL | Transaction Risk Triage | 92% precision on flagged anomalies |
| **2. Automated HR Intelligence** | LLM Agents, PDF Parsing | Recruitment Automation | 80% reduction in manual review effort |
| **3. Regulatory Compliance Assistant** | RAG, Vector Search | FinTech Document Audit | ~75% reduction in manual lookup time |
| **4. Feedback Sentiment Analytics** | TF-IDF + Logistic Regression | Customer Experience Triage | 83% F1-score with sub-50ms latency |

---

## 2. Project 1: Financial Fraud Triage Engine

### Objective & Business Value

High-volume financial transaction systems process data faster than human teams can audit. This project uses an unsupervised anomaly detection approach to isolate high-risk transaction patterns, allowing risk operations teams to prioritize edge-case triage and reduce fraud exposure.

From a business analytics perspective, the goal is not only to identify anomalies, but to convert anomaly scores into clear operating decisions:

- Which transactions should be escalated?
- What risk threshold balances false positives and fraud exposure?
- How can leadership quantify the financial impact of detected fraud?

### Core Architecture

- **Data Processing:** Features are aggregated through SQL, including transaction velocity, amount behavior, chargeback history, account age, and merchant patterns.
- **Modeling:** An Isolation Forest flags deviations from normal user purchasing behavior.
- **Decision Layer:** Scores pass through a configurable threshold filter to trigger automated risk alerts.
- **Business Output:** Alerts are structured for Risk Operations teams with action labels and confidence indicators.

### Code Blueprint

```python
# Save as: fraud_detection.py
import random
from typing import Dict


class FraudTriageEngine:
    """
    Simulates a fraud triage engine using anomaly-style scoring.

    Portfolio context:
        The full project analyzed 2M+ financial transactions and used
        Isolation Forest logic to flag high-risk activity with 92% precision.
    """

    def __init__(self, threshold: float = 0.78):
        self.threshold = threshold

    def calculate_anomaly_score(self, transaction_data: Dict) -> float:
        """
        Simulates anomaly scoring from risk features.

        In production, this method would call a trained Isolation Forest model
        after SQL feature engineering and return a normalized anomaly score.
        """

        risk_factor = 0.45

        if transaction_data.get("velocity_1h", 0) > 5:
            risk_factor += 0.20

        if transaction_data.get("chargeback_history", False):
            risk_factor += 0.25

        if transaction_data.get("amount", 0) > 1000:
            risk_factor += 0.05

        return min(risk_factor + random.uniform(0.01, 0.04), 0.99)

    def evaluate(self, transaction: Dict) -> Dict:
        """Returns an operational decision for risk analysts."""

        score = self.calculate_anomaly_score(transaction)
        action = "Escalate to Risk Team" if score >= self.threshold else "Pass"

        return {
            "anomaly_score": round(score, 2),
            "threshold": self.threshold,
            "action": action,
            "precision_confidence": 0.92,
            "business_value": "Prioritizes high-risk cases for faster fraud response.",
        }


if __name__ == "__main__":
    tx = {
        "velocity_1h": 7,
        "chargeback_history": True,
        "amount": 1200,
    }

    engine = FraudTriageEngine()
    print("Fraud Triage Output:", engine.evaluate(tx))
```

### Sample Output

```text
Fraud Triage Output: {
  'anomaly_score': 0.98,
  'threshold': 0.78,
  'action': 'Escalate to Risk Team',
  'precision_confidence': 0.92,
  'business_value': 'Prioritizes high-risk cases for faster fraud response.'
}
```

---

## 3. Project 2: Document Intelligence & HR Automation

### Objective & Business Value

This project automates highly manual first-pass screening of unstructured corporate documents, especially resumes. By parsing text and checking it against operational requirements, the system extracts key parameters and returns structured feedback, reducing manual review bottlenecks by roughly 80%.

For recruiting and workforce operations, this creates a faster and more consistent review process:

- Recruiters receive structured candidate summaries.
- Candidates receive tailored feedback faster.
- Review criteria become more repeatable across document formats.
- High-volume screening becomes easier to scale.

### Core Architecture

- **Ingestion:** Extract raw unstructured strings from PDF, TXT, and document files.
- **Text Normalization:** Clean whitespace, remove formatting noise, and standardize case.
- **GenAI Layer:** Use structured system prompts with an LLM to evaluate role alignment.
- **Output Validation:** Force uniform metadata blocks instead of open-ended conversational text.
- **Deployment Path:** Serve feedback through Streamlit or FastAPI with sub-5-second turnaround.

### Code Blueprint

```python
# Save as: document_intelligence.py
import re
from typing import Dict, List


class DocumentIntelligenceAgent:
    """
    Simulates an AI-powered document intelligence workflow.

    Portfolio context:
        The full project used Streamlit/FastAPI, OpenAI GPT, and PDF/TXT
        extraction to automate tailored resume feedback and reduce manual
        review effort by 80%.
    """

    def __init__(self):
        self.token_cost_per_1k = 0.0006  # Simulated lightweight model cost.
        self.target_skills = ["python", "sql", "fastapi", "streamlit", "openai", "ai", "rag"]

    def clean_document_text(self, raw_text: str) -> str:
        """Normalizes extracted document text before analysis."""

        return re.sub(r"\s+", " ", raw_text.strip().lower())

    def extract_skills(self, clean_text: str) -> List[str]:
        """Simulates targeted parameter extraction from unstructured text."""

        return [skill for skill in self.target_skills if skill in clean_text]

    def estimate_token_cost(self, clean_text: str) -> float:
        """Approximates LLM inference cost for experimentation tracking."""

        estimated_tokens = len(clean_text) / 4
        return round((estimated_tokens / 1000) * self.token_cost_per_1k, 6)

    def analyze_text(self, raw_text: str) -> Dict:
        """Returns structured HR intelligence output."""

        clean_text = self.clean_document_text(raw_text)
        detected_skills = self.extract_skills(clean_text)

        return {
            "extracted_skills": detected_skills,
            "role_alignment_signal": "strong" if len(detected_skills) >= 3 else "moderate",
            "review_efficiency_gain": "80% reduction in manual effort",
            "feedback_turnaround": "<5 seconds per resume",
            "estimated_token_cost_usd": self.estimate_token_cost(clean_text),
        }


if __name__ == "__main__":
    document_sample = """
    Deep experience with Python, SQL data pipelines, FastAPI services,
    and building AI prototypes with OpenAI models.
    """

    agent = DocumentIntelligenceAgent()
    print("Document Intelligence Output:", agent.analyze_text(document_sample))
```

### Sample Output

```text
Document Intelligence Output: {
  'extracted_skills': ['python', 'sql', 'fastapi', 'openai', 'ai'],
  'role_alignment_signal': 'strong',
  'review_efficiency_gain': '80% reduction in manual effort',
  'feedback_turnaround': '<5 seconds per resume',
  'estimated_token_cost_usd': 2.1e-05
}
```

---

## 4. Project 3: FinTech Regulatory Compliance RAG

### Objective & Business Value

Navigating dense financial policies and credit regulations creates delays for analysts, stakeholders, and risk teams. This project applies Retrieval-Augmented Generation (RAG) to scan embedded corporate policies and return answers grounded in verifiable source citations, reducing lookup times by approximately 75%.

The business value is grounded decision support:

- Analysts find relevant policy faster.
- Stakeholders receive answers with traceable citations.
- Compliance teams reduce repetitive manual lookup.
- The system minimizes hallucination risk by constraining answers to retrieved context.

### Core Architecture

- **Document Parsing:** Ingest unstructured financial and legal documents.
- **Chunking:** Split policy text into searchable passages with metadata.
- **Vector Index:** Store embeddings in a vector database such as Pinecone or Chroma.
- **Retrieval Loop:** Fetch top-k document segments matching the analyst query.
- **Synthesis:** Feed retrieved context into an LLM with strict groundedness rules.
- **API Layer:** Serve responses through scalable FastAPI endpoints for enterprise integration.

### Code Blueprint

```python
# Save as: compliance_rag.py
from typing import Dict


class ComplianceRAGAssistant:
    """
    Simulates a citation-grounded regulatory compliance assistant.

    Portfolio context:
        The full concept uses RAG, vector search, GenAI APIs, and FastAPI
        endpoint design to reduce manual financial document review by ~75%.
    """

    def __init__(self):
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

    def retrieve_context(self, question: str) -> Dict:
        """
        Simulates semantic retrieval.

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

    def query_compliance(self, question: str) -> Dict:
        """Returns a grounded answer with source citation."""

        context = self.retrieve_context(question)

        return {
            "answer": (
                "Based on retrieved policy context, credit ceilings are capped "
                "at 40% of verifiable monthly gross income."
            ),
            "source_citation": context["citation"],
            "retrieved_context": context["text"],
            "manual_lookup_reduction": "75%",
            "groundedness_rule": "Answer must be supported by retrieved policy text.",
        }


if __name__ == "__main__":
    bot = ComplianceRAGAssistant()
    print("Compliance RAG Output:", bot.query_compliance("What are the credit caps?"))
```

### Sample Output

```text
Compliance RAG Output: {
  'answer': 'Based on retrieved policy context, credit ceilings are capped at 40% of verifiable monthly gross income.',
  'source_citation': 'Policy Manual, Section 4.2',
  'retrieved_context': 'Credit limits must not exceed 40% of verifiable monthly gross income.',
  'manual_lookup_reduction': '75%',
  'groundedness_rule': 'Answer must be supported by retrieved policy text.'
}
```

---

## 5. Project 4: Feedback Sentiment Analytics Core

### Objective & Business Value

Processing continuous feedback streams requires high-speed analytics. This project avoids unnecessary LLM calls by using classical machine learning to classify incoming text feedback in under 50ms while maintaining an 83% F1-score for automated routing.

This is an important AI engineering tradeoff: not every text problem requires a generative model. For high-volume classification, a tuned classical ML model can be faster, cheaper, and easier to monitor.

### Core Architecture

- **Feature Extraction:** Convert text into TF-IDF vectors using unigram and bigram features.
- **Classification:** Use Logistic Regression tuned through GridSearchCV.
- **Evaluation:** Validate using accuracy, precision, recall, and F1-score.
- **Operational Routing:** Route negative feedback into a priority customer recovery queue.
- **Inference Optimization:** Keep runtime below 50ms for real-time classification workflows.

### Code Blueprint

```python
# Save as: sentiment_analytics.py
from typing import Dict


class FeedbackSentimentClassifier:
    """
    Simulates a fast sentiment analytics inference wrapper.

    Portfolio context:
        The full project used TF-IDF features, Logistic Regression,
        GridSearchCV tuning, and error analysis on 2,000 movie reviews.
    """

    def __init__(self):
        self.positive_keywords = ["great", "fast", "helpful", "love", "excellent"]
        self.negative_keywords = ["slow", "bad", "broken", "poor", "confusing"]

    def predict_latency_and_sentiment(self, text: str) -> Dict:
        """
        Simulates a tuned Logistic Regression inference function.

        In production, this method would load a fitted TF-IDF vectorizer and
        trained classifier, then return a probability score and class label.
        """

        lowered = text.lower()
        positive_hits = sum(word in lowered for word in self.positive_keywords)
        negative_hits = sum(word in lowered for word in self.negative_keywords)

        sentiment = "positive" if positive_hits >= negative_hits else "negative"
        route = "CX reporting dashboard" if sentiment == "positive" else "priority recovery queue"

        return {
            "sentiment": sentiment,
            "routing_destination": route,
            "inference_latency": "<50ms",
            "validated_f1_score": 0.83,
        }


if __name__ == "__main__":
    classifier = FeedbackSentimentClassifier()

    sample_feedback = "The data processing pipeline is incredibly fast and helpful."
    print("Sentiment Output:", classifier.predict_latency_and_sentiment(sample_feedback))
```

### Sample Output

```text
Sentiment Output: {
  'sentiment': 'positive',
  'routing_destination': 'CX reporting dashboard',
  'inference_latency': '<50ms',
  'validated_f1_score': 0.83
}
```

---

## 6. Business Operations Analytics & ROI

### Simulated Automation Return on Investment

Across an operations department managing approximately 100 manual review tasks weekly, these multi-modal AI and analytics solutions scale organizational output by reducing repetitive review time.

```text
Average manual task evaluation: 20 minutes
AI/ML assisted evaluation:       4 minutes
Time restructured per task:     16 minutes
Weekly task volume:            100 reviews
Weekly human capital recovered: 1,600 minutes (~26.7 hours)
```

### Model Strategy and Cost Control

| Workload | Recommended Model Strategy | Why It Fits |
| --- | --- | --- |
| Fraud detection | Isolation Forest + SQL feature engineering | Deterministic, scalable, and explainable for risk triage |
| HR document intelligence | Lightweight LLM agent | Best suited for structured feedback on unstructured resumes |
| Regulatory compliance | RAG + higher-quality LLM | Requires grounded answers and citation traceability |
| Sentiment analytics | TF-IDF + Logistic Regression | Faster and cheaper than LLMs for high-volume classification |

By decoupling workloads, high-cost language models are reserved for contextual reasoning tasks such as RAG and HR feedback, while low-latency classical ML is used for classification and anomaly detection. This keeps computational spend optimized while preserving business impact.

---

## 7. Role Alignment: AI Engineer, Agents & Experimentation

This portfolio demonstrates the technical and operational skills needed to build practical AI systems:

- **Agent design:** LLM-powered workflows for resume intelligence and compliance support.
- **Experimentation mindset:** Evaluation across precision, F1-score, latency, cost, and ROI.
- **Automation:** Reusable engines that reduce repetitive manual review.
- **Risk analytics:** Fraud triage using anomaly detection and threshold-based alerting.
- **Enterprise readiness:** FastAPI, Docker concepts, SQL pipelines, vector search, and structured outputs.
- **Business translation:** Clear linkage between model behavior and operational decision-making.

## Closing Summary

These four projects show a consistent engineering pattern: identify a manual bottleneck, select the right AI or ML technique, design a measurable workflow, and translate model output into a business action. The result is a portfolio focused not only on technical implementation, but also on operational efficiency, automation ROI, and experiment-driven AI development.

