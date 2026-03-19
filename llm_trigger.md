# Context: VocalMind - AI-Powered Customer Service Evaluation System
You are an expert backend engineer and AI researcher. You are helping draft the core Python/FastAPI backend modules for "VocalMind," a graduation project focused on Explainable AI (XAI) in customer service evaluation. 

you need to inspect our database first to understand.

## Tech Stack
- Backend: Python, FastAPI
- LLM Orchestration: LangChain
- LLM Provider: Groq (using Llama-3)
- Vector Database: we have to add to our existing
- Observability: LangSmith

## Task Objective
Draft the implementation for three advanced evaluation modules. These must go beyond standard API calls and implement novel DSAI (Data Science and AI) research contributions: Cross-Modal Dissonance, Dialogue State Tracking, and NLI-based Evaluation.

Please generate the Pydantic models, the LangChain prompts (using Few-Shot examples where appropriate), and the core pipeline functions for the following three features.

---

### Feature 1: Emotion Shift "Why?" (Cross-Modal Dissonance & Counterfactuals)
**Goal:** Detect when the acoustic emotion contradicts the text sentiment, explain the root cause, and generate an actionable correction.

**Requirements:**
1. Create a Pydantic model `EmotionShiftAnalysis` with fields:
   - `is_dissonance_detected` (bool)
   - `dissonance_type` (str: e.g., "Sarcasm", "Passive-Aggression", or "None")
   - `root_cause` (str: grounded in the transcript)
   - `counterfactual_correction` (str: starting with "If the agent had...")
2. Create a LangChain `ChatPromptTemplate` that accepts `agent_context`, `customer_text`, and `acoustic_emotion`.
3. The prompt must instruct the LLM to act as a behavioral analyst and specifically look for contradictions between text and audio.
4. Draft a function `analyze_emotion_shift()` that chains the prompt, LLM, and structured output parser.

---

### Feature 2: Topic Detection & Process Adherence (Dialogue State Tracking)
**Goal:** Identify the topic, retrieve the standard operating procedure (SOP), and evaluate if the agent followed the optimal resolution path, not just if they solved it.

**Requirements:**
1. Create a Pydantic model `ProcessAdherenceReport` with fields:
   - `detected_topic` (str)
   - `is_resolved` (bool)
   - `efficiency_score` (int: 1-10)
   - `missing_sop_steps` (list of strings)
2. Draft a function `evaluate_process_adherence(transcript_text, retrieved_sop_from_pinecone)`.
3. The prompt should ask the LLM to map the transcript against the `retrieved_sop_from_pinecone` and identify any steps the agent skipped or handled inefficiently.

---

### Feature 3: Answer Similarity to Ground Truth (NLI-Based Evaluation)
**Goal:** Replace basic cosine similarity with Natural Language Inference (NLI) to differentiate between conversational variations and actual policy hallucinations.

**Requirements:**
1. Create a Pydantic model `NLIEvaluation` with fields:
   - `nli_category` (Literal: "Entailment", "Benign Deviation", "Contradiction", "Policy Hallucination")
   - `justification` (str)
2. Draft a function `run_nli_policy_check(agent_statement, ground_truth_policy)`.
3. The system prompt must clearly define the four categories:
   - **Entailment:** Agent statement is fully supported.
   - **Benign Deviation:** Agent added small talk/empathy not in the policy (Do not penalize).
   - **Contradiction:** Agent directly violated the policy.
   - **Policy Hallucination:** Agent invented a rule not found in the ground truth.

---

## Output Instructions 
- Write clean, fully typed Python code.
- Ensure all LangChain chains use the modern LCEL (LangChain Expression Language) syntax (e.g., `chain = prompt | model | parser`).
- Assume the LangSmith environment variables are already set in the environment for tracing.
- Provide mock implementations for database retrievals if necessary, focusing the core logic on the LLM orchestration.