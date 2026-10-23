"""Temporary gold-quote diagnostic used while authoring eval/gold.jsonl."""

import asyncio

from sqlalchemy import text

from docqa.answer import normalise_text
from docqa.config import get_settings
from docqa.db import make_engine
from docqa.eval import load_gold

TARGETS = ("B2", "B3", "S1", "S2", "D2", "D4", "D5")


async def main() -> None:
    settings = get_settings()
    engine = make_engine(settings)
    questions = {question.id: question for question in load_gold()}
    async with engine.connect() as conn:
        for question_id in TARGETS:
            gold = questions[question_id].gold
            assert gold is not None
            result = await conn.execute(
                text(
                    "SELECT c.text FROM chunks AS c JOIN documents AS d "
                    "ON d.id=c.document_id WHERE d.doc_id=:doc "
                    "AND c.chunk_index=:idx"
                ),
                {"doc": gold.doc_id, "idx": gold.chunk_index},
            )
            chunk_text = str(result.scalar_one())
            quote = normalise_text(gold.quote)
            body = normalise_text(chunk_text)
            print(question_id, "found", quote in body)
            if quote in body:
                continue
            low, high = 0, len(quote)
            while low < high:
                middle = (low + high + 1) // 2
                if quote[:middle] in body:
                    low = middle
                else:
                    high = middle - 1
            print("  matched prefix tail:", repr(quote[:low][-90:]))
            print("  next quote chars:  ", repr(quote[low : low + 70]))
            index = body.find(quote[:low][-40:])
            print("  db around:         ", repr(body[index : index + 200]))
    await engine.dispose()


asyncio.run(main())
