"""The production retrieve-then-answer path shared by the CLI commands.

``docqa eval`` keeps its own harness (it runs the retrieval ablation for every
question), but ``docqa ask`` and ``docqa warm`` need exactly one hybrid+rerank
retrieval and one validated answer per question. Building that here avoids two
copies drifting apart.
"""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from docqa.answer import (
    AnswerResponse,
    Calibration,
    StructuredModel,
    answer_question,
    build_model,
)
from docqa.config import Settings
from docqa.db import make_engine, make_session_factory, session_scope
from docqa.ingest.embedder import Embedder, build_embedder
from docqa.rerank import Reranker, build_reranker
from docqa.retrieval import (
    PgCandidateSource,
    RetrievalTrace,
    retrieve_with_trace,
)


class ProductionAnswerer:
    """One configured embedder/reranker/model over the ingested corpus."""

    def __init__(
        self,
        settings: Settings,
        calibration: Calibration,
        *,
        k: int = 8,
        force_generation: bool = False,
    ) -> None:
        """Build the collaborators; the database is contacted per question."""
        embedder_choice = build_embedder(settings)
        reranker_choice = build_reranker(
            mode=os.environ.get("DOCQA_RERANKER", "auto").strip().lower(),
            cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
        )
        self.embedder_banner = embedder_choice.banner
        self.reranker_banner = reranker_choice.banner
        self.model: StructuredModel = build_model(settings)
        self.calibration = calibration
        self.k = k
        self.force_generation = force_generation
        self._embedder: Embedder = embedder_choice.embedder
        self._reranker: Reranker = reranker_choice.reranker
        self._engine: AsyncEngine = make_engine(settings)
        self._factory: async_sessionmaker[AsyncSession] = make_session_factory(
            self._engine
        )

    async def answer(
        self, question: str
    ) -> tuple[AnswerResponse, RetrievalTrace]:
        """Retrieve with the production path, then validate one answer."""
        async with session_scope(self._factory) as session:
            result = await retrieve_with_trace(
                question,
                self.k,
                embedder=self._embedder,
                source=PgCandidateSource(session),
                reranker=self._reranker,
            )
            response = await answer_question(
                question,
                result.chunks,
                model=self.model,
                calibration=self.calibration,
                force_generation=self.force_generation,
            )
            return response, result.trace

    async def close(self) -> None:
        """Dispose the database engine."""
        await self._engine.dispose()
