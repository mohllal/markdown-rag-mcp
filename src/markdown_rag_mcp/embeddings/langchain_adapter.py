"""
LangChain embedding adapter.

Adapts our HuggingFace embedder to LangChain's Embeddings interface
for seamless integration with langchain-milvus.
"""

import asyncio
import logging

import nest_asyncio
from langchain_core.embeddings import Embeddings

from markdown_rag_mcp.config.settings import RAGConfig
from markdown_rag_mcp.embeddings.embedder import HuggingFaceEmbedder

logger = logging.getLogger(__name__)


class LangChainEmbeddingAdapter(Embeddings):
    """
    Adapter to make our HuggingFaceEmbedder compatible with LangChain.

    Wraps our async embedding provider to provide LangChain's expected
    synchronous interface.
    """

    def __init__(self, config: RAGConfig):
        """Initialize the adapter with our HuggingFace embedder."""
        self.config = config
        self._embedder = HuggingFaceEmbedder(config)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the underlying embedder."""
        if not self._initialized:
            # HuggingFaceEmbedder uses lazy loading, no explicit initialization needed
            # Just ensure the model loads by accessing it
            _ = self._embedder.model  # This will trigger model loading
            self._initialized = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents (synchronous interface for LangChain).

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        # Run async method in synchronous context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If we're in an async context, we need to use a different approach
            # This is a common issue with LangChain sync/async mixing
            try:
                nest_asyncio.apply()
            except ImportError:
                # If nest_asyncio is not available, try to handle gracefully
                logger.warning("nest_asyncio not available, may have issues with async/sync mixing")

        return loop.run_until_complete(self._embedder.generate_batch_embeddings(texts))

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text (synchronous interface for LangChain).

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        # Run async method in synchronous context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If we're in an async context, we need to use a different approach
            try:
                nest_asyncio.apply()
            except ImportError:
                # If nest_asyncio is not available, try to handle gracefully
                logger.warning("nest_asyncio not available, may have issues with async/sync mixing")

        return loop.run_until_complete(self._embedder.generate_embedding(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Async version of embed_documents (LangChain's async interface).

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        return await self._embedder.generate_batch_embeddings(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """
        Async version of embed_query (LangChain's async interface).

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

        return await self._embedder.generate_embedding(text)

    @property
    def embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        return self._embedder.embedding_dimension

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.embedding_model
