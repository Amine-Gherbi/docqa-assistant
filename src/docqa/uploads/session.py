"""Opaque client identity: a random id inside an HMAC-signed cookie.

There is no account and no email. The id is only an ownership key; the HMAC is
what makes a forged cookie useless. A missing or invalid cookie always mints a
new id on ``GET /session`` and never authorises an existing upload.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import uuid

CLIENT_COOKIE = "docqa_client"
CLIENT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
COOKIE_MAX_AGE_S = 30 * 24 * 60 * 60


class SessionSigner:
    """Mint and verify the opaque client id."""

    def __init__(self, secret: str) -> None:
        """Bind the signing secret; an empty secret is a configuration error."""
        if not secret:
            raise ValueError("a session signing secret is required")
        self._secret = secret.encode("utf-8")

    def _signature(self, client_id: str) -> str:
        """HMAC-SHA256 of the id, hex encoded."""
        return hmac.new(
            self._secret, client_id.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def token_for(self, client_id: str) -> str:
        """Sign an existing id."""
        return f"{client_id}.{self._signature(client_id)}"

    def new_token(self) -> tuple[str, str]:
        """Return ``(client_id, cookie_token)`` for a fresh random id."""
        client_id = uuid.uuid4().hex
        return client_id, self.token_for(client_id)

    def verify(self, token: str | None) -> str | None:
        """Return the signed client id, or ``None`` for anything else."""
        if not token:
            return None
        client_id, separator, signature = token.partition(".")
        if not separator or not CLIENT_ID_RE.fullmatch(client_id):
            return None
        if not hmac.compare_digest(signature, self._signature(client_id)):
            return None
        return client_id

    def session_for(self, token: str | None) -> tuple[str, bool]:
        """Keep a valid session, otherwise mint one; reports whether it is new."""
        client_id = self.verify(token)
        if client_id is not None:
            return client_id, False
        fresh, _ = self.new_token()
        return fresh, True
