from ragas import evaluate
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from rag_app.config import settings

# Setup
try:
    groq_llm = ChatGroq(
        model=settings.groq.model,
        api_key=settings.groq.api_key.get_secret_value(),
        temperature=0.0,
    )
    evaluator_llm = LangchainLLMWrapper(langchain_llm=groq_llm)

    ollama_embeddings = OllamaEmbeddings(
        base_url=settings.embedding.base_url,
        model=settings.embedding.model,
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings=ollama_embeddings)

    # Dummy Data
    sample = SingleTurnSample(
        user_input="What is the capital of France?",
        response="The capital of France is Paris.",
        retrieved_contexts=["Paris is the capital and most populous city of France."],
        reference="Paris",
    )

    dataset = EvaluationDataset(samples=[sample])

    print("Running Ragas evaluation on dummy data...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("Result:", result)
    print("Scores:", result.to_pandas().iloc[0].to_dict())

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
