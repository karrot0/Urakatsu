"""Small shared value types."""

from __future__ import annotations

from pydantic import BaseModel


class Availability(BaseModel):
    ok: bool
    reason: str = ""

    @classmethod
    def yes(cls) -> "Availability":
        return cls(ok=True)

    @classmethod
    def no(cls, reason: str) -> "Availability":
        return cls(ok=False, reason=reason)
