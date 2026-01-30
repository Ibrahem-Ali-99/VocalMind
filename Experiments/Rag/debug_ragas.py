try:
    from ragas.llms import LangchainLLM  # noqa: F401

    print("LangchainLLM found")
except ImportError as e:
    print(f"LangchainLLM NOT found: {e}")

try:
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: F401

    print("LangchainEmbeddingsWrapper found")
except ImportError as e:
    print(f"LangchainEmbeddingsWrapper NOT found: {e}")
