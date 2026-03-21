import os
import sys
import traceback

p = os.path.abspath(os.path.join(os.getcwd(), '..', 'services'))
sys.path.append(p)


def main() -> None:
    from rag.query_engine import RAGQueryEngine

    try:
        engine = RAGQueryEngine()
        print("Engine initialized")
        result = engine.query_answer(question='hello', org_filter=None)
        print("Result:", result)
    except Exception:
        print("EXCEPTION OCCURRED:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
