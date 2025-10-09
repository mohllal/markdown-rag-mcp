# Tasks: Markdown RAG System (Core Library)

**Input**: Design documents from `/specs/001-markdown-rag-system/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: TDD workflow enforced - tests MUST be written first and FAIL before implementation begins per constitutional requirement III.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Core library in `src/markdown_rag_mcp/`
- Test structure: `tests/unit/`, `tests/integration/`, `tests/contract/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure per implementation plan with directories: `src/markdown_rag_mcp/`, `tests/`, `docker/`, `examples/`
- [x] T002 Initialize Python 3.12.0 project with pyproject.toml and core dependencies (LangChain, HuggingFaceEmbeddings, python-frontmatter, Milvus)
- [x] T003 [P] Configure linting and formatting tools (ruff, black)
- [x] T004 [P] Create Docker Compose configuration in `docker/docker-compose.yml` for Milvus ecosystem (etcd, MinIO, Milvus)
- [x] T005 [P] Create development Dockerfile in `docker/Dockerfile.dev` for Python 3.12.0 environment

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create abstract interfaces in `src/markdown_rag_mcp/core/interfaces.py` for IRAGEngine, IVectorStore, IEmbeddingProvider, IDocumentParser
- [x] T007 [P] Create base data models in `src/markdown_rag_mcp/models/__init__.py` for Document, DocumentSection, QueryResult
- [x] T008 [P] Implement configuration management in `src/markdown_rag_mcp/config/settings.py` with environment variable support
- [x] T009 [P] Create error handling classes in `src/markdown_rag_mcp/models/exceptions.py` for MarkdownRAGError, MilvusConnectionError, EmbeddingModelError
- [x] T010 Setup Milvus connection and collection management in `src/markdown_rag_mcp/storage/base_store.py`
- [x] T011 Initialize main library exports in `src/markdown_rag_mcp/__init__.py` exposing core interfaces

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Query Relevant Documentation (Priority: P1) 🎯 MVP

**Goal**: Enable semantic search across markdown files with natural language queries, returning relevant sections with confidence scores >0.7

**Independent Test**: Load sample markdown files, run query "authentication setup", verify relevant sections returned with confidence scores

### Tests for User Story 1 (TDD - Write Tests First) ⚠️

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T012 [P] [US1] Contract test for RAGEngine.search() interface in `tests/contract/test_core_interfaces/test_rag_engine.py`
- [ ] T013 [P] [US1] Contract test for CLI search command in `tests/contract/test_cli_contracts/test_search_command.py`
- [ ] T014 [P] [US1] Integration test for end-to-end query workflow in `tests/integration/test_end_to_end/test_basic_search.py`
- [ ] T015 [P] [US1] Unit tests for markdown parsing in `tests/unit/test_parsers/test_markdown_parser.py`
- [ ] T016 [P] [US1] Unit tests for document chunking in `tests/unit/test_indexing/test_chunker.py`
- [ ] T017 [P] [US1] Unit tests for embedding generation in `tests/unit/test_indexing/test_embedder.py`
- [ ] T018 [P] [US1] Unit tests for vector storage in `tests/unit/test_storage/test_milvus_store.py`
- [ ] T019 [P] [US1] Unit tests for similarity search in `tests/unit/test_search/test_similarity_matcher.py`

### Implementation for User Story 1

- [x] T020 [P] [US1] Implement markdown parser in `src/markdown_rag_mcp/parsers/markdown_parser.py` using markdown-it-py
- [x] T021 [P] [US1] Implement document models in `src/markdown_rag_mcp/models/document.py` for Document and DocumentSection entities
- [x] T022 [P] [US1] Implement query models in `src/markdown_rag_mcp/models/query.py` for QueryResult and SearchRequest entities
- [ ] T023 [US1] Implement document chunker in `src/markdown_rag_mcp/indexing/chunker.py` with heading-based boundaries and size limits
- [x] T024 [US1] Implement HuggingFace embedder in `src/markdown_rag_mcp/indexing/embedder.py` using sentence-transformers/all-MiniLM-L6-v2
- [x] T025 [US1] Implement Milvus vector store in `src/markdown_rag_mcp/storage/milvus_store.py` with collection management
- [ ] T026 [US1] Implement indexer orchestrator in `src/markdown_rag_mcp/indexing/indexer.py` coordinating parsing, chunking, and embedding
- [x] T027 [US1] Implement query processor in `src/markdown_rag_mcp/search/query_processor.py` for natural language query handling
- [ ] T028 [US1] Implement similarity matcher in `src/markdown_rag_mcp/search/similarity_matcher.py` with >0.7 threshold filtering
- [x] T029 [US1] Implement main RAG engine in `src/markdown_rag_mcp/core/rag_engine.py` orchestrating all components
- [x] T030 [US1] Implement CLI search command in `src/markdown_rag_mcp/cli/commands.py` with JSON and human-readable output
- [x] T031 [US1] Implement CLI main entry point in `src/markdown_rag_mcp/cli/main.py` with argument parsing
- [ ] T032 [US1] Add comprehensive error handling and logging for User Story 1 operations
- [ ] T033 [US1] Create CLI index command in `src/markdown_rag_mcp/cli/commands.py` for manual directory indexing

**Checkpoint**: At this point, User Story 1 should be fully functional - basic semantic search with CLI interface

---

## Phase 4: User Story 2 - Frontmatter-Enhanced Search (Priority: P2)

**Goal**: Utilize optional frontmatter metadata to improve search accuracy and relevance scoring

**Independent Test**: Create markdown files with frontmatter (title, tags, llm_hints), verify improved relevance vs files without frontmatter

### Tests for User Story 2 (TDD - Write Tests First) ⚠️

- [ ] T034 [P] [US2] Contract test for frontmatter parsing in `tests/contract/test_core_interfaces/test_frontmatter_parsing.py`
- [ ] T035 [P] [US2] Integration test for frontmatter-enhanced search in `tests/integration/test_end_to_end/test_frontmatter_search.py`
- [ ] T036 [P] [US2] Unit tests for frontmatter parser in `tests/unit/test_parsers/test_frontmatter_parser.py`
- [ ] T037 [P] [US2] Unit tests for metadata enhancement in `tests/unit/test_indexing/test_metadata_enhancer.py`

### Implementation for User Story 2

- [ ] T038 [P] [US2] Implement frontmatter parser in `src/markdown_rag_mcp/parsers/frontmatter_parser.py` using python-frontmatter
- [ ] T039 [P] [US2] Extend document models in `src/markdown_rag_mcp/models/document.py` to include frontmatter metadata
- [ ] T040 [US2] Implement metadata enhancer in `src/markdown_rag_mcp/indexing/metadata_enhancer.py` for frontmatter integration
- [ ] T041 [US2] Extend chunker in `src/markdown_rag_mcp/indexing/chunker.py` to incorporate frontmatter context
- [ ] T042 [US2] Extend embedder in `src/markdown_rag_mcp/indexing/embedder.py` to enhance embeddings with metadata
- [ ] T043 [US2] Update Milvus store in `src/markdown_rag_mcp/storage/milvus_store.py` to handle metadata fields
- [ ] T044 [US2] Update similarity matcher in `src/markdown_rag_mcp/search/similarity_matcher.py` for metadata-enhanced ranking
- [ ] T045 [US2] Integrate frontmatter processing into RAG engine workflow
- [ ] T046 [US2] Update CLI to support metadata inclusion in search results with --include-metadata flag

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - enhanced search with optional frontmatter

---

## Phase 5: User Story 3 - Incremental Index Updates (Priority: P3)

**Goal**: Automatically detect file changes and update search index without full re-indexing

**Independent Test**: Add new markdown file, verify searchable without manual re-indexing; modify file, verify updates reflected

### Tests for User Story 3 (TDD - Write Tests First) ⚠️

- [ ] T047 [P] [US3] Contract test for file monitoring in `tests/contract/test_core_interfaces/test_file_monitoring.py`
- [ ] T048 [P] [US3] Integration test for incremental updates in `tests/integration/test_end_to_end/test_incremental_indexing.py`
- [ ] T049 [P] [US3] Unit tests for file watcher in `tests/unit/test_monitoring/test_file_watcher.py`
- [ ] T050 [P] [US3] Unit tests for change detection in `tests/unit/test_indexing/test_change_detector.py`

### Implementation for User Story 3

- [ ] T051 [P] [US3] Implement file watcher in `src/markdown_rag_mcp/monitoring/file_watcher.py` for directory monitoring
- [ ] T052 [P] [US3] Implement change detector in `src/markdown_rag_mcp/indexing/change_detector.py` using file hashes
- [ ] T053 [US3] Extend document models in `src/markdown_rag_mcp/models/document.py` to track indexing timestamps and file hashes
- [ ] T054 [US3] Implement incremental indexer in `src/markdown_rag_mcp/indexing/incremental_indexer.py` for selective updates
- [ ] T055 [US3] Extend Milvus store in `src/markdown_rag_mcp/storage/milvus_store.py` for document deletion and updates
- [ ] T056 [US3] Update RAG engine in `src/markdown_rag_mcp/core/rag_engine.py` to support monitoring mode
- [ ] T057 [US3] Add CLI watch command in `src/markdown_rag_mcp/cli/commands.py` with --watch flag for continuous monitoring
- [ ] T058 [US3] Implement monitoring coordinator for handling file events and triggering appropriate index updates

**Checkpoint**: All user stories should now be independently functional - full RAG system with automatic updates

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and production readiness

- [ ] T059 [P] Create comprehensive library documentation in `README.md` with usage examples
- [ ] T060 [P] Create library usage examples in `examples/basic_usage.py`, `examples/advanced_queries.py`, `examples/custom_extensions.py`
- [ ] T061 [P] Implement CLI status command in `src/markdown_rag_mcp/cli/commands.py` for system health and statistics
- [ ] T062 [P] Implement CLI config command in `src/markdown_rag_mcp/cli/commands.py` for configuration management
- [ ] T063 [P] Add comprehensive logging configuration across all components
- [ ] T064 [P] Performance optimization for large document collections (batch processing, connection pooling)
- [ ] T065 [P] Add error recovery mechanisms for Milvus connection failures
- [ ] T066 [P] Security hardening for file system operations and input validation
- [ ] T067 Run quickstart.md validation to ensure end-to-end functionality
- [ ] T068 [P] Create packaging configuration for library distribution
- [ ] T069 [P] Create accuracy measurement framework in `tests/integration/test_accuracy_validation.py` with ground truth dataset and 85% accuracy validation for SC-002
- [ ] T070 [P] Generate ground truth query-answer pairs for technical domain accuracy testing in `tests/fixtures/ground_truth_dataset.json`
- [ ] T071 [P] Implement Milvus performance optimization in `src/markdown_rag_mcp/storage/milvus_store.py` with connection pooling, batch operations, and index tuning
- [ ] T072 [P] Add performance benchmarking suite in `tests/performance/test_scale_limits.py` to validate SC-001, SC-004, and SC-006 metrics

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Extends US1 components but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Extends US1/US2 components but independently testable

### Within Each User Story

- Tests MUST be written first and FAIL before implementation (TDD requirement)
- Models before services
- Services before orchestration
- Core implementation before CLI integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models and parsers within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (TDD - write first):
Task: "Contract test for RAGEngine.search() interface in tests/contract/test_core_interfaces/test_rag_engine.py"
Task: "Integration test for end-to-end query workflow in tests/integration/test_end_to_end/test_basic_search.py"
Task: "Unit tests for markdown parsing in tests/unit/test_parsers/test_markdown_parser.py"

# Launch all models for User Story 1 together:
Task: "Implement document models in src/markdown_rag_mcp/models/document.py"
Task: "Implement query models in src/markdown_rag_mcp/models/query.py"
Task: "Implement markdown parser in src/markdown_rag_mcp/parsers/markdown_parser.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Basic semantic search)
4. **STOP and VALIDATE**: Test User Story 1 independently using quickstart.md scenarios
5. Deploy/demo core RAG functionality

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP - Basic semantic search!)
3. Add User Story 2 → Test independently → Deploy/Demo (Enhanced search with frontmatter)
4. Add User Story 3 → Test independently → Deploy/Demo (Automatic index updates)
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Core search functionality)
   - Developer B: User Story 2 (Frontmatter enhancement)
   - Developer C: User Story 3 (Incremental updates)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- **TDD CRITICAL**: Verify tests fail before implementing (constitutional requirement)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Focus on library-first architecture - CLI serves as reference implementation
- Maintain >0.7 similarity threshold and 85% accuracy targets
- Support up to 1000 markdown files and 10GB collections
- **Total Tasks: 73** (Updated with accuracy measurement and performance optimization tasks)
