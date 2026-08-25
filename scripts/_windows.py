"""Temporary anchor-window printer used while authoring eval/gold.jsonl."""

import asyncio

from sqlalchemy import text

from docqa.answer import normalise_text
from docqa.config import get_settings
from docqa.db import make_engine
from docqa.eval import load_gold

SPECS = (
    ("S1", "fine-tune"),
    ("S1", "softmax"),
    ("D2", "inner product"),
    ("D2", "simpler"),
)


async def main() -> None:
    settings = get_settings()
    engine = make_engine(settings)
    questions = {question.id: question for question in load_gold()}
    async with engine.connect() as conn:
        for question_id, anchor in SPECS:
            gold = questions[question_id].gold
            assert gold is not None
            result = await conn.execute(
                text(
                    "SELECT c.text FROM chunks AS c JOIN documents AS d "
                    "ON d.id=c.document_id WHERE d.doc_id=:doc AND "
                    "c.chunk_index=:idx"
                ),
                {"doc": gold.doc_id, "idx": gold.chunk_index},
            )
            body = normalise_text(str(result.scalar_one()))
            index = body.find(anchor.lower())
            print(f"\n{question_id} / {anchor!r} at {index}")
            print(repr(body[max(0, index - 120) : index + 240]))


asyncio.run(main())
