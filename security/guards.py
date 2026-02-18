import hashlib
import hmac
import os
import re
import time
from typing import List


_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(previous|all|above|prior)\s+instructions?",
    r"system\s*prompt",
    r"you\s+are\s+now",
    r"pretend\s+(to\s+be|you('re|\s+are))",
    r"act\s+as\s+(?!.*character)",
    r"override\s+(your|all|safety)",
    r"jailbreak",
    r"disregard\s+(your|all|any)",
    r"new\s+instruction",
    r"from\s+now\s+on\s+(you|ignore|act)",
    r"<\s*script",
    r"<\s*iframe",
    r"javascript\s*:",
    r"data\s*:\s*text",
    r"base64\s*,",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]

_SECRET: bytes = os.environ.get(
    "HMAC_SECRET",
    os.urandom(32).hex()
).encode("utf-8")


def sanitize(text: str, max_length: int = 4096) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    text = text[:max_length]

    for pattern in _COMPILED:
        if pattern.search(text):
            raise ValueError("Potentially malicious content detected and blocked")

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def sign_payload(data: str) -> str:
    return hmac.new(_SECRET, data.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_payload(data: str, signature: str) -> bool:
    expected = sign_payload(data)
    return hmac.compare_digest(expected, signature)


class RateLimiter:
    def __init__(self, max_calls: int = 20, window_seconds: int = 60):
        self._calls: dict = {}
        self.max_calls = max_calls
        self.window = window_seconds

    def allow(self, key: str) -> bool:
        now = time.time()
        history: List[float] = [
            t for t in self._calls.get(key, [])
            if now - t < self.window
        ]
        if len(history) >= self.max_calls:
            self._calls[key] = history
            return False
        history.append(now)
        self._calls[key] = history
        return True


_global_limiter = RateLimiter(max_calls=20, window_seconds=60)


def check_rate(agent_id: str) -> bool:
    return _global_limiter.allow(agent_id)
