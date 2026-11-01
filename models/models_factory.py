# Abstract factory imports
from abc import ABC, abstractmethod
# Ollama factory imports
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding # Base class for embeddings
from llama_index.llms.ollama import Ollama
from llama_index.core.llms.llm import LLM # Base class for LLMs

class AbstractModelFactory(ABC):
    """
    Abstract base class for all models.
    """
    SEED = 42

    @abstractmethod
    def obtain_embeddings_model(self):
        """
        Abstract method to obtain the embeddings model.
        """
        pass

    @abstractmethod
    def obtain_LLM_model(self):
        """
        Abstract method to obtain the LLM model.
        """
        pass

class LlamaIndexModelFactory(AbstractModelFactory):
    """
    Model factory for LlamaIndex agents.
    """
    URL_MODELS = "http://localhost:11434"

    def obtain_embeddings_model(self) -> BaseEmbedding:
        """
        Obtain the embeddings model.
        """
        model_embeddings = OllamaEmbedding(
            model_name = "nomic-embed-text",
            base_url = self.URL_MODELS
        )
        return model_embeddings

    def obtain_LLM_model(self) -> LLM:
        """
        Obtain the LLM model.
        """
        model_llm = Ollama(
            model="gpt-oss:20b",
            base_url=self.URL_MODELS,
            request_timeout=180.0,
            context_window=8192,
            temperature=0.0, # Deterministic output
            thinking = False,
            additional_kwargs={"seed": self.SEED} # Reproducible results
        )
        return model_llm