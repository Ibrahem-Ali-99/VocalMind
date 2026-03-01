import json
import argparse
import os
import re
from typing import Optional, Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv

# --- Utilities ---

def get_llm_client() -> BaseChatModel:
    """Configures and returns the Groq LLM client."""
    # Try loading .env from the script's directory if not already set
    if not os.getenv("GROQ_API_KEY"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY incomplete. Please set it in your environment or .env file.")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=api_key,
        temperature=0.0,
    )

def parse_json_response(content: str) -> Dict[str, Any]:
    """Robustly parses JSON from LLM output, handling markdown blocks."""
    content = content.strip()
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        content = match.group(1) if match else content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)

def read_content(source: str) -> str:
    """Reads content from a file path or returns the string directly."""
    if os.path.exists(source):
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            pass # Treat as raw text if read fails
    return source

def generate_rag_answer(question: str) -> str:
    """Generates an answer using the RAG system (lazy load)."""
    print("Initializing RAG system (this may take a moment)...")
    try:
        # Lazy import to keep the script lightweight when not using RAG
        import sys

        # Ensure rag_app is importable.
        # Assuming script is in Experiments/Rag and rag_app is in Experiments/Rag/rag_app
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.append(script_dir)

        from rag_app.pipeline import DocumentIngestionPipeline
        from rag_app.query_engine import RAGQueryEngine

        # Initialize pipeline (loads index)
        pipeline = DocumentIngestionPipeline()
        index = pipeline.run(force_reindex=False)

        # Query
        engine = RAGQueryEngine(index)
        response = engine.query(question)
        return str(response)

    except ImportError as e:
        return f"Error loading RAG system: {e}. Make sure you are in the correct environment."
    except Exception as e:
        return f"Error during RAG generation: {e}"

# --- Evaluators ---

class AnswerComparator:
    """Compares two answers using an LLM as an impartial judge."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def compare(self, answer1: str, answer2: str, question: Optional[str] = None) -> Dict[str, Any]:
        system_prompt = """You are an impartial judge evaluating semantic similarity.
        Goal: Determine if Answer A and Answer B convey the same core information.
        Guidelines:
        - Focus on meaning, ignoring minor phrasing differences.
        - Output JSON: {{"similarity_score": 1-10, "explanation": "reason"}}
        """

        user_prompt = """
        Question: {question}
        Answer A: {answer1}
        Answer B: {answer2}
        """

        chain = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)]) | self.llm
        print(f"Comparing inputs (len A={len(answer1)}, len B={len(answer2)})...")

        try:
            return parse_json_response(chain.invoke({
                "question": question or "N/A",
                "answer1": answer1,
                "answer2": answer2
            }).content)
        except Exception as e:
            return {"similarity_score": 0, "explanation": f"Error: {e}"}

class ResolutionEvaluator:
    """Analyzes a conversation transcript for problem resolution."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def evaluate(self, conversation: List[Dict]) -> Dict[str, Any]:
        transcript_lines = []
        for t in conversation:
            speaker = t.get('speaker', 'Unknown').upper()
            emotion = t.get('emotion')
            emotion_str = f" ({emotion})" if emotion else ""
            text = t.get('text', '')
            transcript_lines.append(f"{speaker}{emotion_str}: {text}")
        transcript = "\n".join(transcript_lines)

        system_prompt = """You are a Quality Assurance Specialist.
        Goal: Determine if the Agent resolved the Client's problem.
        match the JSON format:
        {{
            "solved": boolean,
            "problem_summary": "string",
            "resolution_summary": "string",
            "evidence": "string"
        }}
        """

        user_prompt = "Analyze this transcript:\n\n{transcript_text}"

        chain = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)]) | self.llm
        print(f"Analyzing transcript ({len(conversation)} turns)...")

        try:
            return parse_json_response(chain.invoke({"transcript_text": transcript}).content)
        except Exception as e:
            return {"solved": False, "problem_summary": "Error", "resolution_summary": str(e), "evidence": "N/A"}

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Tools")
    subparsers = parser.add_subparsers(dest="command")

    # Compare Mode
    cmd_compare = subparsers.add_parser("compare", help="Compare similarity")
    cmd_compare.add_argument("answer1", nargs="?", help="Answer to evaluate (or first answer)")
    cmd_compare.add_argument("answer2", nargs="?", help="Reference answer (optional if using --rag)")
    cmd_compare.add_argument("-q", "--question", help="Question text (required for --rag)")
    cmd_compare.add_argument("-i", "--interactive", action="store_true")
    cmd_compare.add_argument("--rag", action="store_true", help="Generate Answer 2 using RAG system")

    # Analyze Mode
    cmd_analyze = subparsers.add_parser("analyze", help="Analyze resolution")
    cmd_analyze.add_argument("file", help="Path to conversation JSON")

    args = parser.parse_args()

    # Default to compare if ambiguous
    if not args.command:
        args.command = "compare"

    try:
        llm = get_llm_client()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    if args.command == "compare":
        if args.rag:
            if not args.question:
                print("Error: --rag requires a --question to query the system.")
                return
            if not args.answer1:
                 print("Error: --rag requires an answer1 to compare against the RAG system.")
                 return

            # Generate reference answer from RAG
            a1 = read_content(args.answer1)
            q = args.question
            print(f"Querying RAG system with: '{q}'...")
            a2 = generate_rag_answer(q)
            print(f"\nRAG Answer: {a2}\n")

        elif args.interactive or not (args.answer1 and args.answer2):
            print("\n=== Interactive Comparator ===")
            q = input("Question (optional): ").strip()
            a1 = input("Answer A: ").strip()
            a2 = input("Answer B: ").strip()
            print("\nThinking...")
        else:
            q, a1, a2 = args.question, read_content(args.answer1), read_content(args.answer2)

        res = AnswerComparator(llm).compare(a1, a2, q)
        print(f"\nSCORE: {res.get('similarity_score')}/10\nREASON: {res.get('explanation')}\n")

    elif args.command == "analyze":
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return

        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Detect if it's a single conversation (list of dicts) or multiple (list of lists)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            conversations = data
        else:
            conversations = [data]

        evaluator = ResolutionEvaluator(llm)

        print(f"\nrunning analysis on {len(conversations)} conversation(s)...\n")

        for i, conv in enumerate(conversations):
            print(f"--- Conversation {i+1} ---")
            res = evaluator.evaluate(conv)
            print(f"SOLVED:      {'YES' if res.get('solved') else 'NO'}")
            print(f"PROBLEM:     {res.get('problem_summary')}")
            print(f"RESOLUTION:  {res.get('resolution_summary')}")
            print(f"EVIDENCE:    \"{res.get('evidence')}\"\n")

if __name__ == "__main__":
    main()
