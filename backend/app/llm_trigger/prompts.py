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
- evidence_quotes: includes at least one verbatim quote from customer_text
- citations: includes structured quote entries with source="transcript"

Example 2:
Input:
- customer_text: "Okay, do whatever you want."
- acoustic_emotion: "disgust"
Output style:
- is_dissonance_detected: true
- dissonance_type: "Passive-Aggression"
- root_cause: references resignation language with hostile tone
- counterfactual_correction: starts with "If the agent had..."
- evidence_quotes: includes at least one verbatim quote from customer_text
- citations: includes structured quote entries with source="transcript"
""".strip()


NLI_FEW_SHOT = """
Example A:
- ground_truth_policy: "Refunds are allowed only within 30 days."
- agent_statement: "I can help process a refund if your purchase is within 30 days."
- nli_category: "Entailment"
- evidence_quotes: includes one quote from policy and one from agent statement

Example B:
- ground_truth_policy: "Refunds are allowed only within 30 days."
- agent_statement: "No worries, I totally understand your frustration. Let me check your purchase date first."
- nli_category: "Benign Deviation"
- evidence_quotes: includes one quote from policy and one from agent statement

Example C:
- ground_truth_policy: "Refunds are allowed only within 30 days."
- agent_statement: "We always allow refunds up to 90 days."
- nli_category: "Contradiction"
- evidence_quotes: includes one quote from policy and one from agent statement

Example D:
- ground_truth_policy: "Refunds are allowed only within 30 days."
- agent_statement: "Policy says refunds require manager approval plus a processing fee."
- nli_category: "Policy Hallucination"
- evidence_quotes: includes one quote from policy and one from agent statement
""".strip()


def build_emotion_shift_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a behavioral analyst for customer-service QA. "
                "Detect cross-modal contradictions between text and acoustic emotion. "
                "Ground all claims in the provided text and keep output valid JSON only. "
                "Every claim must cite verbatim evidence in double quotes. "
                "Always populate both evidence_quotes and citations fields.\n\n"
                "DOCUMENT GOVERNANCE:\n"
                "- Policy documents define mandatory rules and compliance.\n"
                "- SOPs define operational procedures.\n"
                "- When policy and SOP appear to conflict, policy always takes precedence.\n"
                "- Do NOT treat policy or SOP as free-form knowledge.\n"
                "- Use them ONLY as supporting context when the transcript clearly shows a procedural issue explaining an emotion shift.\n"
                "- If evidence is insufficient to explain a shift, return 'insufficient evidence' in root_cause.\n"
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
                "3) Explain root cause grounded in transcript text. If no clear cause, return 'insufficient evidence'.\n"
                "4) Provide a correction sentence that starts exactly with 'If the agent had...'.\n"
                "5) Include evidence_quotes as verbatim excerpts from customer text.\n"
                "6) Include citations with source, speaker, quote, and utterance_index when known.",
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
                "All justifications must include verbatim quote-grounded evidence. "
                "Always populate evidence_quotes and citations. "
                "Return strict JSON only.\n\n"
                "DOCUMENT GOVERNANCE:\n"
                "- SOP documents define the operational procedure.\n"
                "- Policy constraints override SOP when both are present.\n"
                "- Evaluate process adherence, escalation flow, and verification steps based on SOP.\n"
                "- If evidence is insufficient to verify a step, mark as missing or return 'insufficient evidence' in justification.\n"
                "{format_instructions}",
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
                "- Write a short justification paragraph explicitly referencing the transcript evidence.\n"
                "- Compare transcript trajectory against expected graph/SOP and list missing steps.\n"
                "- List missed SOP steps precisely as short bullet-style strings.\n"
                "- Provide evidence_quotes using exact transcript snippets in double quotes.\n"
                "- Provide citations with transcript/SOP quote provenance.",
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
                "- Policy Hallucination: invented rule not present in policy.\n\n"
                "DOCUMENT GOVERNANCE:\n"
                "- Policy documents define mandatory rules and compliance requirements.\n"
                "- Policy documents are the PRIMARY source of truth.\n"
                "- If evidence is insufficient to confirm a match or violation, return 'insufficient evidence' in justification.\n"
                "Justification must be quote-grounded with exact excerpts. "
                "Always include evidence_quotes and citations.\n"
                "Return strict JSON only.\n{format_instructions}",
            ),
            (
                "human",
                "{few_shot}\n\n"
                "Ground truth policy:\n{ground_truth_policy}\n\n"
                "Agent statement:\n{agent_statement}\n\n"
                "Classify into one category and justify with textual evidence.\n"
                "Include at least one policy quote and one agent quote in evidence_quotes and citations.",
            ),
        ]
    )
