"""The client cookie is opaque and signed; a forged value is not a client."""

from __future__ import annotations

from docqa.uploads.session import SessionSigner


def test_a_minted_token_verifies_to_its_client_id() -> None:
    signer = SessionSigner("test-secret")
    client_id, token = signer.new_token()
    assert len(client_id) == 32
    assert token.startswith(f"{client_id}.")
    assert signer.verify(token) == client_id


def test_a_tampered_signature_is_rejected() -> None:
    signer = SessionSigner("test-secret")
    client_id, _token = signer.new_token()
    forged = f"{client_id}.{'0' * 64}"
    assert signer.verify(forged) is None


def test_a_token_signed_with_another_secret_is_rejected() -> None:
    first = SessionSigner("first-secret")
    second = SessionSigner("second-secret")
    _client_id, token = first.new_token()
    assert second.verify(token) is None


def test_missing_and_malformed_tokens_do_not_mint_a_client() -> None:
    signer = SessionSigner("test-secret")
    assert signer.verify(None) is None
    assert signer.verify("") is None
    assert signer.verify("not-a-token") is None
    assert signer.verify("../etc/passwd") is None
    assert signer.verify("f" * 32 + ".") is None


def test_session_for_keeps_a_valid_client_and_mints_otherwise() -> None:
    signer = SessionSigner("test-secret")
    client_id, token = signer.new_token()
    assert signer.session_for(token) == (client_id, False)
    fresh, is_new = signer.session_for(None)
    assert is_new is True
    assert fresh != client_id
