"""Smoke-test ``POST /ask`` against the real local corpus.

Requires the R1 database and the cached ONNX models. With no
``DOCQA_MODEL_BASE_URL`` configured it exercises the labelled extractive
fallback, the real retriever, and the committed calibration end to end.

    uv run python scripts/smoke_ask.py
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from docqa.api import create_app

QUESTIONS = [
    "How many attention heads does the Transformer use?",
    "What is masked language modelling used for in BERT?",
    "How does dense passage retrieval differ from BM25?",
]


def main() -> int:
    """Post each question and print the validated claims and their citations."""
    app = create_app()
    with TestClient(app) as client:
        for question in QUESTIONS:
            response = client.post("/ask", json={"question": question})
            body = response.json()
            print(f"Q: {question}")
            print(
                f"  status={response.status_code} "
                f"refused={body['insufficient_evidence']} "
                f"reason={body['reason']} p={body['refusal_probability']}"
            )
            print(f"  model={body['model']}")
            for claim in body["claims"][:2]:
                citation = claim["citations"][0]
                print(f"  claim: {claim['text'][:120]}")
                print(
                    f"  cite:  {citation['doc_id']} p.{citation['page_no']} "
                    f"sha={citation['page_text_sha256'][:12]} "
                    f"quote={citation['quote'][:90]!r}"
                )
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
