# import asyncio
# import math
# import time
# from typing import Any
# from datasets import Dataset
# from google import genai
# from langchain_core.documents import Document
# from ragas import evaluate
# from ragas.llms import llm_factory
# from ragas.embeddings import GoogleEmbeddings
# from ragas.metrics import faithfulness,answer_relevancy


# from app.config import get_settings
# from app.utils.logger import get_logger

# logger=get_logger(__name__)

# class RAGASEvaluator:

#     def __init__(self):

#         logger.info("Initializing RAGAS evaluator")
#         self.settings= get_settings()

#         eval_llm_model=self.settings.ragas_llm_model or "gemini-2.5-flash-lite"
#         eval_llm_temperature= (self.settings.ragas_llm_temperature if self.settings.ragas_llm_temperature is not None else 0.0)
#         eval_embedding_model = self.settings.ragas_embedding_model or "models/gemini-embedding-001"

#         self.gemini_client= genai.Client(api_key=self.settings.gemini_api_key) 

#         self.llm=llm_factory(
#             model=eval_llm_model,
#             provider='google',
#             client=self.gemini_client,
#             temperature=eval_llm_temperature
#         )

#         self.embedding=GoogleEmbeddings(
#             model=eval_embedding_model,
#             client=self.gemini_client
#         )

#         self.metrics = [faithfulness,answer_relevancy]

#         logger.info(f"RAGAS Evaluator initialized- "
#                     f"LLM: {eval_llm_model} and temp: {eval_llm_temperature} "
#                     f"Embedding: {eval_embedding_model} "
#                     f"Metrics : {[metric.name for metric in self.metrics]}"
#         )

#     async def aevaluate(self,question: str, answer: str, contexts: list[str]) -> dict[str,Any]:
#         logger.debug(f"Starting evaluation for the question: {question[:100]}...")
#         start_time=time.time()

#         try:

#             dataset= self._prepare_dataset(question,answer,contexts)

#             results= await asyncio.to_thread(self._evaluate_with_timeout, dataset)

#             scores = {
#                 "faithfulness": self._sanitize_score(results.get("faithfulness")),
#                 "answer_relevancy": self._sanitize_score(results.get("answer_relevancy")),
#                 "evaluation_time": (time.time() - start_time) * 1000,
#                 "error": None
#             }

#             if self.settings.ragas_log_results:
#                 logger.info(
#                     f"Evaluation Completed - "
#                     f"faithfulness={scores['faithfulness']}, "
#                     f"answer_relevancy={scores['answer_relevancy']}, "
#                     f"time={scores['evaluation_time']}ms"
#                 )

#             return scores
        
#         except Exception as e:
#             logger.warning(f"Evaluation Failed: {e}",exc_info=True)
#             return self._handle_evaluation_errors(e)


#     def _prepare_dataset(self,question: str, answer: str, contexts:list[str]) -> Dataset:

#         data={
#             "question":[question],
#             "answer":[answer],
#             "contexts":[contexts]
#         }

#         logger.debug(f"Prepared dataset with {len(contexts)} contexts for question: {question[:50]}...")
#         return Dataset.from_dict(data)
    
#     def _evaluate_with_timeout(self,dataset: Dataset) -> dict[str,Any]:
#         result=evaluate(
#             dataset=dataset,
#             metrics=self.metrics,
#             llm=self.llm,
#             embeddings=self.embedding
#         )

#         return result.to_pandas().to_dict("records")[0]
    
#     def _handle_evaluation_errors(self,error:Exception)->dict[str,Any]:
#         logger.error(f"Returning fallback scores due to error: {error}")
#         return{
#             "faithfulness":None,
#             "answer_relevancy":None,
#             "evaluation_time_ms": None,
#             "error": str(error)
#         }
    
#     @staticmethod
#     def _sanitize_score(value):
#         """Convert NaN floats to None to satisfy Pydantic validation."""
#         if value is None:
#             return None
#         if isinstance(value, float) and math.isnan(value):
#             return None
#         return value
import math
import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, aevaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.run_config import RunConfig

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:

    def __init__(self):
        logger.info("Initializing RAGAS evaluator")
        self.settings = get_settings()

        eval_llm_model = self.settings.ragas_llm_model or "gemini-2.5-flash-lite"
        eval_llm_temperature = (
            self.settings.ragas_llm_temperature
            if self.settings.ragas_llm_temperature is not None
            else 0.0
        )
        eval_embedding_model = (
            self.settings.ragas_embedding_model or "models/text-embedding-004"
        )

        # ← Use LangchainLLMWrapper instead of llm_factory with genai client
        self.llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=eval_llm_model,
                temperature=eval_llm_temperature,
                google_api_key=self.settings.gemini_api_key,
                request_timeout=120,
            )
        )

        # ← Use LangchainEmbeddingsWrapper instead of GoogleEmbeddings
        self.embedding = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(
                model=eval_embedding_model,
                google_api_key=self.settings.gemini_api_key,
            )
        )

        self.metrics = [Faithfulness(),ResponseRelevancy()]

        run_config = RunConfig()
        for metric in self.metrics:
            if isinstance(metric, MetricWithLLM):
                metric.llm = self.llm
            if isinstance(metric, MetricWithEmbeddings):
                metric.embeddings = self.embedding
            metric.init(run_config)

        logger.info(
            f"RAGAS Evaluator initialized - "
            f"LLM: {eval_llm_model} and temp: {eval_llm_temperature} "
            f"Embedding: {eval_embedding_model} "
            f"Metrics: {[metric.name for metric in self.metrics]}"
        )

    async def aevaluate(
        self, question: str, answer: str, contexts: list[str]
    ) -> dict[str, Any]:
        logger.debug(f"Starting evaluation for the question: {question[:100]}...")
        start_time = time.time()

        try:
            dataset = self._prepare_dataset(question, answer, contexts)

            results = await self._evaluate_async(dataset)

            scores = {
                "faithfulness": self._sanitize_score(results.get("faithfulness")),
                "answer_relevancy": self._sanitize_score(results.get("response_relevancy")),
                "evaluation_time_ms": (time.time() - start_time) * 1000,
                "error": None,
            }

            if self.settings.ragas_log_results:
                logger.info(
                    f"Evaluation Completed - "
                    f"faithfulness={scores['faithfulness']}, "
                    f"answer_relevancy={scores['answer_relevancy']}, "
                    f"time={scores['evaluation_time_ms']}ms"
                )

            return scores

        except Exception as e:
            logger.warning(f"Evaluation Failed: {e}", exc_info=True)
            return self._handle_evaluation_errors(e)

    def _prepare_dataset(
        self, question: str, answer: str, contexts: list[str]
    ) -> EvaluationDataset:                                      # ← returns EvaluationDataset now
        sample = SingleTurnSample(
            user_input=question,                                  # ← was "question"
            response=answer,                                      # ← was "answer"
            retrieved_contexts=contexts,                          # ← was "contexts"
        )
        logger.debug(
            f"Prepared dataset with {len(contexts)} contexts for question: {question[:50]}..."
        )
        return EvaluationDataset(samples=[sample])

    async def _evaluate_async(self, dataset: EvaluationDataset) -> dict[str, Any]:
        result = await aevaluate(
            dataset=dataset,
            metrics=self.metrics,
            run_config=RunConfig(
            timeout=120,        # per-call timeout in seconds
            max_retries=2,      # reduce retries (default is higher)
            max_wait=60,        # max wait between retries
            ),
            raise_exceptions=False #Temporarily added (remove later)
        )
        # return result.to_pandas().to_dict("records")[0]
        df = result.to_pandas()
        logger.info(f"Raw ragas result: {df.to_dict('records')}")  # ← log raw output
        return df.to_dict("records")[0]
    
    def _handle_evaluation_errors(self, error: Exception) -> dict[str, Any]:
        logger.error(f"Returning fallback scores due to error: {error}")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "evaluation_time_ms": None,
            "error": str(error),
        }

    @staticmethod
    def _sanitize_score(value):
        """Convert NaN floats to None to satisfy Pydantic validation."""
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return value