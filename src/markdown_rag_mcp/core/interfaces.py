"""
Abstract interfaces for the Markdown RAG system.

These interfaces define the contracts for core components, enabling extensibility
and dependency injection for testing and future implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from src.markdown_rag_mcp.models.document import Document, DocumentSection
from src.markdown_rag_mcp.models.query import QueryResult


class IDocumentParser(ABC):
    """Interface for parsing different document formats."""

    @abstractmethod
    async def parse_file(self, file_path: Path) -> Document:
        """
        Parse a file and return a Document object.

        Args:
            file_path: Path to the file to parse

        Returns:
            Document object with parsed content and metadata

        Raises:
            DocumentParsingError: If the file cannot be parsed
        """
        pass

    @abstractmethod
    def supports_file_type(self, file_path: Path) -> bool:
        """
        Check if this parser supports the given file type.

        Args:
            file_path: Path to check

        Returns:
            True if the parser can handle this file type
        """
        pass


class IEmbeddingProvider(ABC):
    """Interface for generating embeddings from text."""

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingModelError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingModelError: If batch embedding generation fails
        """
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name/identifier of the embedding model."""
        pass


class IVectorStore(ABC):
    """Interface for vector database operations."""

    @abstractmethod
    async def initialize_collections(self) -> None:
        """
        Initialize required database collections/indexes.

        Raises:
            VectorStoreError: If initialization fails
        """
        pass

    @abstractmethod
    async def store_document_sections(self, sections: list[DocumentSection], embeddings: list[list[float]]) -> None:
        """
        Store document sections with their embeddings.

        Args:
            sections: Document sections to store
            embeddings: Corresponding embedding vectors

        Raises:
            VectorStoreError: If storage operation fails
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """
        Search for similar document sections.

        Args:
            query_embedding: Query vector to search for
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            metadata_filters: Optional metadata filtering criteria

        Returns:
            List of matching results with similarity scores

        Raises:
            VectorStoreError: If search operation fails
        """
        pass

    @abstractmethod
    async def delete_document(self, file_path: str) -> None:
        """
        Delete all sections belonging to a specific document.

        Args:
            file_path: Path of the document to delete

        Raises:
            VectorStoreError: If deletion fails
        """
        pass

    @abstractmethod
    async def update_document_sections(
        self, file_path: str, sections: list[DocumentSection], embeddings: list[list[float]]
    ) -> None:
        """
        Update sections for a document (delete old, insert new).

        Args:
            file_path: Path of the document to update
            sections: New document sections
            embeddings: Corresponding embedding vectors

        Raises:
            VectorStoreError: If update operation fails
        """
        pass

    @abstractmethod
    async def get_section_count(self) -> int:
        """Get the total number of indexed sections."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check the health and status of the vector store.

        Returns:
            Dictionary with health status information
        """
        pass


class IRAGEngine(ABC):
    """
    Main interface for the RAG engine, orchestrating all components.

    This is the primary interface that external systems (CLI, APIs, MCP servers)
    should use to interact with the RAG functionality.
    """

    @abstractmethod
    async def index_directory(
        self,
        directory_path: Path,
        recursive: bool = True,
        force_reindex: bool = False,
        file_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Index all supported documents in a directory.

        Args:
            directory_path: Path to directory to index
            recursive: Whether to index subdirectories
            force_reindex: Whether to reindex existing documents
            file_patterns: Optional list of glob patterns to filter files

        Returns:
            Dictionary with indexing results and statistics

        Raises:
            IndexingError: If indexing fails
        """
        pass

    @abstractmethod
    async def index_file(self, file_path: Path, force_reindex: bool = False) -> dict[str, Any]:
        """
        Index a single document file.

        Args:
            file_path: Path to the file to index
            force_reindex: Whether to reindex if already exists

        Returns:
            Dictionary with indexing results

        Raises:
            IndexingError: If indexing fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        include_metadata: bool = False,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """
        Search for documents matching the query.

        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            include_metadata: Whether to include document metadata
            metadata_filters: Optional metadata filtering criteria

        Returns:
            List of matching results ordered by relevance

        Raises:
            SearchError: If search fails
        """
        pass

    @abstractmethod
    async def start_monitoring(
        self, directory_path: Path, recursive: bool = True, file_patterns: list[str] | None = None
    ) -> None:
        """
        Start monitoring a directory for file changes.

        Args:
            directory_path: Path to monitor
            recursive: Whether to monitor subdirectories
            file_patterns: Optional list of glob patterns to filter files

        Raises:
            MonitoringError: If monitoring cannot be started
        """
        pass

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """
        Stop file monitoring if currently active.

        Raises:
            MonitoringError: If monitoring cannot be stopped
        """
        pass

    @abstractmethod
    async def get_status(self) -> dict[str, Any]:
        """
        Get current system status and statistics.

        Returns:
            Dictionary with system status information
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the RAG engine and all its components.

        Raises:
            InitializationError: If initialization fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the RAG engine and cleanup resources.

        Raises:
            ShutdownError: If shutdown fails
        """
        pass


class IFileMonitor(ABC):
    """Interface for monitoring file system changes."""

    @abstractmethod
    async def start_watching(self, path: Path, recursive: bool = True, patterns: list[str] | None = None) -> None:
        """
        Start monitoring a path for changes.

        Args:
            path: Path to monitor
            recursive: Whether to monitor subdirectories
            patterns: Optional glob patterns to filter files

        Raises:
            MonitoringError: If monitoring cannot be started
        """
        pass

    @abstractmethod
    async def stop_watching(self) -> None:
        """
        Stop monitoring and cleanup resources.

        Raises:
            MonitoringError: If monitoring cannot be stopped
        """
        pass

    @abstractmethod
    def get_change_events(self) -> AsyncIterator[dict[str, Any]]:
        """
        Get an async iterator of file change events.

        Yields:
            Dictionary containing event type and file path
        """
        pass

    @property
    @abstractmethod
    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active."""
        pass
