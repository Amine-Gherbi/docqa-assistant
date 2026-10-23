"""Temporary corpus-span finder used while authoring eval/gold.jsonl."""

import asyncio
import sys

from sqlalchemy import text

from docqa.config import get_settings
from docqa.db import make_engine


async def main() -> None:
    settings = get_settings()
    engine = make_engine(settings)
    async with engine.connect() as conn:
        for phrase in sys.argv[1:]:
            result = await conn.execute(
                text(
                    "SELECT d.doc_id, c.chunk_index, c.page_no, "
                    "regexp_replace(c.text, '\\s+', ' ', 'g') AS t "
                    "FROM chunks c JOIN documents d ON d.id=c.document_id "
                    "WHERE regexp_replace(c.text, '\\s+', ' ', 'g') "
                    "ILIKE :p ORDER BY d.doc_id, c.chunk_index"
                ),
                {"p": f"%{phrase}%"},
            )
            rows = result.all()
            print(f"\n=== {phrase!r}: {len(rows)} hit(s)")
            for doc, index, page, body in rows[:3]:
                at = body.lower().find(phrase.lower())
                lo, hi = max(0, at - 110), min(len(body), at + len(phrase) + 170)
                print(f"  {doc} #{index} p{page}: ...{body[lo:hi]}...")
    await engine.dispose()


asyncio.run(main())
