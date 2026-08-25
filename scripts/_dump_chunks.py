"""Temporary chunk dumper used while authoring eval/gold.jsonl."""

import asyncio
import sys

from sqlalchemy import text

from docqa.config import get_settings
from docqa.db import make_engine


async def main() -> None:
    settings = get_settings()
    engine = make_engine(settings)
    async with engine.connect() as conn:
        for spec in sys.argv[1:]:
            doc_id, raw_index = spec.rsplit("#", 1)
            result = await conn.execute(
                text(
                    "SELECT c.page_no, regexp_replace(c.text, '\\s+', ' ', 'g') AS t "
                    "FROM chunks c JOIN documents d ON d.id=c.document_id "
                    "WHERE d.doc_id = :doc AND c.chunk_index = :idx"
                ),
                {"doc": doc_id, "idx": int(raw_index)},
            )
            row = result.one_or_none()
            if row is None:
                print(f"\n### {spec}: MISSING")
                continue
            page, body = row
            print(f"\n### {doc_id} #{raw_index} p{page} ({len(body)} chars)")
            print(body[:2400])
    await engine.dispose()


asyncio.run(main())
