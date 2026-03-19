from langchain_core.prompts import ChatPromptTemplate


EMOTION_SHIFT_FEW_SHOT = """
Example 1:
Input:
- customer_text: "I am thrilled this happened again, amazing service."
- acoustic_emotion: "anger"
Output style:
- is_dissonance_detected: true
- dissonance_type: "Sarcasm"
- root_cause: references lexical positivity with angry prosody
- counterfactual_correction: starts with "If the agent had..."

Example 2:
Input:
- customer_text: "Okay, do whatever you want."
- acoustic_emotion: "disgust"
Output style:
- is_dissonance_detected: true
- dissonance_type: "Passive-Aggression"
- root_cause: references resignation language with hostile tone
- counterfactual_correction: starts with "If the agent had..."
""".strip()


NLI_FEW_SHOT = """
Example A:
- ground_truth_policy: "Refunds are allowed only within 30 days."
- agent_statement: "I can help process a refund if your purchase is within 30 days."
- nli_category: "Entailment"

Example B:
- ground_truth_policy: "Refunds are allowed only within 30 days."
- agent_statement: "No worries, I totally understand your frustration. Let me check your purchase date first."
- nli_category: "Benign Deviation"

Example C:
- ground_truth_policy: "Refunds are allowed only within 30 days."
- agent_statement: "We always allow refunds up to 90 days."
- nli_category: "Contradiction"

Example D:
- ground_truth_policy: "Refunds are allowed only within 30 days."
- agent_statement: "Policy says refunds require manager approval plus a processing fee."
- nli_category: "Policy Hallucination"
""".strip()


def build_emotion_shift_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a behavioral analyst for customer-service QA. "
                "Detect cross-modal contradictions between text and acoustic emotion. "
                "Ground all claims in the provided text and keep output valid JSON only.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "{few_shot}\n\n"
                "Agent context: {agent_context}\n"
                "Customer text: {customer_text}\n"
                "Acoustic emotion: {acoustic_emotion}\n\n"
                "Task:\n"
                "1) Detect if text sentiment and acoustic emotion are dissonant.\n"
                "2) If dissonant, classify type (e.g., Sarcasm, Passive-Aggression).\n"
                "3) Explain root cause grounded in transcript text.\n"
                "4) Provide a correction sentence that starts exactly with 'If the agent had...'.",
            ),
        ]
    )


def build_process_adherence_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Dialogue State Tracking evaluator. "
                "Map a transcript to the SOP and score process adherence quality, not just outcome. "
                "Return strict JSON only.\n{format_instructions}",
            ),
            (
                "human",
                "Topic hint: {topic_hint}\n\n"
                "Transcript:\n{transcript_text}\n\n"
                "Retrieved SOP:\n{retrieved_sop}\n\n"
                "Expected resolution graph steps:\n{expected_resolution_graph}\n\n"
                "Task:\n"
                "- Detect the primary topic.\n"
                "- Decide if issue is resolved by end of transcript.\n"
                "- Score efficiency from 1-10 considering unnecessary steps, delays, and clarity.\n"
                "- Compare transcript trajectory against expected graph/SOP and list missing steps.\n"
                "- List missed SOP steps precisely as short bullet-style strings.",
            ),
        ]
    )


def build_nli_policy_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an NLI policy evaluator for customer-service QA. "
                "Choose exactly one category:\n"
                "- Entailment: fully supported by policy.\n"
                "- Benign Deviation: empathy/small talk not in policy and not conflicting.\n"
                "- Contradiction: statement violates policy.\n"
                "- Policy Hallucination: invented rule not present in policy.\n"
                "Return strict JSON only.\n{format_instructions}",
            ),
            (
                "human",
                "{few_shot}\n\n"
                "Ground truth policy:\n{ground_truth_policy}\n\n"
                "Agent statement:\n{agent_statement}\n\n"
                "Classify into one category and justify with textual evidence.",
            ),
        ]
    )
