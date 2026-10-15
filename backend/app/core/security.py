"""
API key authentication for state-changing/costly endpoints.

Simple shared-secret check via the X-API-Key header — NOT a full user auth
system (no per-user identity, no roles, no JWT). This exists to stop
anonymous abuse of expensive or destructive endpoints (Groq calls that
cost money, Qdrant writes, full collection wipes) if this API is ever
reachable outside a trusted network — which it is here, since
kubernetes/ingress.yaml exposes /api publicly with no auth layer at all
before this change.

For a real multi-user product (separate accounts, permissions, audit
trail), replace this with OAuth2/JWT. This is intentionally minimal and
matches the current single-operator scope of the project.
"""
import logging

from fastapi import Header, HTTPException, status

from app.core.config import API_KEY

logger = logging.getLogger(__name__)

_warned_once = False


async def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    """
    FastAPI dependency. Add via `Depends(verify_api_key)` on any route that
    writes data, deletes data, or costs money (LLM calls).

    If API_KEY is not set in the environment, auth is effectively disabled
    — this keeps local dev frictionless (no key needed to hit localhost)
    while still requiring one wherever API_KEY is actually configured
    (staging/prod, via kubernetes/secret.yaml or .env).
    """
    global _warned_once

    if not API_KEY:
        if not _warned_once:
            logger.warning(
                "API_KEY is not set — protected endpoints are running "
                "WITHOUT authentication. Set API_KEY in .env (local) or "
                "the backend-auth-secret (Kubernetes) before exposing this "
                "API outside your own machine."
            )
            _warned_once = True
        return

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key (X-API-Key header required)"
        )