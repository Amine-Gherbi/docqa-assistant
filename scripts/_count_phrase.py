"""Temporary per-document phrase counter used while authoring eval/gold.jsonl."""

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
            doc_id, phrase = spec.split("|", 1)
            result = await conn.execute(
                text(
                    "SELECT count(*) FROM chunks c JOIN documents d "
                    "ON d.id=c.document_id WHERE d.doc_id = :doc AND "
                    "regexp_replace(c.text, '\\s+', ' ', 'g') ILIKE :p"
                ),
                {"doc": doc_id, "p": f"%{phrase}%"},
            )
            print(f"{doc_id} / {phrase!r}: {result.scalar_one()} chunk(s)")
    await engine.dispose()


asyncio.run(main())
