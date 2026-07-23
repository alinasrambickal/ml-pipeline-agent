"""
groq_client.py — shared, rate-limited Groq client wrapper.

With 3 branches now running concurrently (one per model, see loop.py), their
Groq calls all draw from the same free-tier key and its single RPM/TPM
budget. This module is the one place all of them go through, so that budget
is respected regardless of how the branches' timing happens to overlap —
the standard client-side throttling pattern for a rate-limited third-party
API (a shared sliding-window limiter), not per-branch hardcoded delays.

It also retries with backoff on RateLimitError / transient API errors, so
hitting the limit degrades gracefully instead of crashing the whole run.
"""

import time
import threading
import collections
import groq
from config import GROQ_API_KEY, GROQ_RPM_LIMIT, GROQ_TPM_LIMIT, GROQ_MAX_RETRIES


class _RateLimiter:
    """Thread-safe sliding-window limiter over requests and tokens per 60s."""

    def __init__(self, max_requests_per_min: int, max_tokens_per_min: int):
        self._lock = threading.Lock()
        self._max_rpm = max_requests_per_min
        self._max_tpm = max_tokens_per_min
        self._calls: collections.deque = collections.deque()  # (timestamp, tokens)

    def acquire(self, estimated_tokens: int) -> None:
        """Blocks until there's room in the current 60s window for one more
        request of roughly this size. Releases the lock while waiting so
        other threads can keep checking in the meantime."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0][0] >= 60:
                    self._calls.popleft()
                used_requests = len(self._calls)
                used_tokens = sum(tokens for _, tokens in self._calls)
                if used_requests < self._max_rpm and used_tokens + estimated_tokens <= self._max_tpm:
                    self._calls.append((now, estimated_tokens))
                    return
                oldest_ts = self._calls[0][0] if self._calls else now
                wait = max(0.1, 60 - (now - oldest_ts))
            time.sleep(wait)


_limiter = _RateLimiter(GROQ_RPM_LIMIT, GROQ_TPM_LIMIT)
_client_lock = threading.Lock()
_client = None


def get_client() -> groq.Groq:
    global _client
    with _client_lock:
        if _client is None:
            _client = groq.Groq(api_key=GROQ_API_KEY)
        return _client


def _estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    """Rough conservative estimate (chars/4, a standard quick approximation)
    plus the reserved output budget — there's no local tokenizer available
    for the served model, so this errs high on purpose."""
    char_count = sum(len(m.get("content", "")) for m in messages)
    return (char_count // 4) + max_tokens


def call_groq(**kwargs):
    """
    Thread-safe, rate-limited, retrying replacement for
    client.chat.completions.create(). Every agent should call this instead
    of hitting the SDK directly, so all branches share one budget.
    """
    client = get_client()
    estimated = _estimate_tokens(kwargs.get("messages", []), kwargs.get("max_tokens", 512))

    last_error = None
    for attempt in range(GROQ_MAX_RETRIES + 1):
        _limiter.acquire(estimated)
        try:
            return client.chat.completions.create(**kwargs)
        except groq.RateLimitError as e:
            last_error = e
            wait = _get_retry_after(e)
            if wait is None:
                wait = 2 ** attempt
            time.sleep(wait)
        except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError) as e:
            last_error = e
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Groq API call failed after {GROQ_MAX_RETRIES + 1} attempts: {last_error}"
    ) from last_error


def _get_retry_after(error: "groq.RateLimitError") -> float | None:
    try:
        value = error.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except Exception:
        return None
