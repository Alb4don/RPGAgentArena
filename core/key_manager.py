import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KeyRecord:
    key: str
    provider: str
    alias: str
    monthly_budget_usd: float = 10.0
    cost_per_1k_input: float = 0.015
    cost_per_1k_output: float = 0.075
    tokens_in: int = 0
    tokens_out: int = 0
    errors_429: int = 0
    errors_5xx: int = 0
    last_used: float = 0.0
    cooldown_until: float = 0.0
    active: bool = True

    def estimated_cost_usd(self) -> float:
        return (
            self.tokens_in / 1000.0 * self.cost_per_1k_input
            + self.tokens_out / 1000.0 * self.cost_per_1k_output
        )

    def budget_remaining(self) -> float:
        return max(0.0, self.monthly_budget_usd - self.estimated_cost_usd())

    def is_available(self) -> bool:
        return (
            self.active
            and time.time() >= self.cooldown_until
            and self.budget_remaining() > 0.001
        )

    def health_score(self) -> float:
        error_penalty = (self.errors_429 * 2 + self.errors_5xx) * 0.05
        budget_factor = min(
            1.0,
            self.budget_remaining() / max(0.001, self.monthly_budget_usd),
        )
        recency_boost = 0.0 if self.last_used == 0.0 else min(
            0.05, 1.0 / max(1.0, time.time() - self.last_used)
        )
        return max(0.0, budget_factor - error_penalty + recency_boost)

    def record_usage(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.last_used = time.time()

    def record_error(self, status: int) -> None:
        if status == 429 or status == 529:
            self.errors_429 += 1
            backoff = min(300.0, 30.0 * (2 ** min(self.errors_429, 4)))
            self.cooldown_until = time.time() + backoff
        elif status >= 500:
            self.errors_5xx += 1
            self.cooldown_until = time.time() + 15.0

    def record_success(self) -> None:
        self.errors_429 = max(0, self.errors_429 - 1)


class KeyManager:
    def __init__(self) -> None:
        self._keys: List[KeyRecord] = []
        self._lock = threading.Lock()
        self._load_from_env()

    def _load_from_env(self) -> None:
        seen: set = set()

        cost_in = float(os.environ.get("COST_INPUT_PER_1K", "0.015"))
        cost_out = float(os.environ.get("COST_OUTPUT_PER_1K", "0.075"))

        primary = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if primary and primary not in seen:
            self._keys.append(KeyRecord(
                key=primary,
                provider="anthropic",
                alias="primary",
                monthly_budget_usd=float(os.environ.get("KEY_BUDGET_USD", "10.0")),
                cost_per_1k_input=cost_in,
                cost_per_1k_output=cost_out,
            ))
            seen.add(primary)

        for i in range(1, 9):
            k = os.environ.get("ANTHROPIC_API_KEY_{}".format(i), "").strip()
            budget = float(os.environ.get("KEY_{}_BUDGET_USD".format(i), "10.0"))
            if k and k not in seen:
                self._keys.append(KeyRecord(
                    key=k,
                    provider="anthropic",
                    alias="key_{}".format(i),
                    monthly_budget_usd=budget,
                    cost_per_1k_input=cost_in,
                    cost_per_1k_output=cost_out,
                ))
                seen.add(k)

        if not self._keys:
            raise EnvironmentError(
                "No API keys found. Set ANTHROPIC_API_KEY in your .env file "
                "or environment variables. Optionally add ANTHROPIC_API_KEY_1 "
                "through ANTHROPIC_API_KEY_8 for automatic key rotation."
            )

    def acquire(self) -> KeyRecord:
        with self._lock:
            available = [k for k in self._keys if k.is_available()]
            if not available:
                active_cooldowns = [
                    k.cooldown_until for k in self._keys
                    if k.active and k.cooldown_until > time.time()
                ]
                if active_cooldowns:
                    wait_s = min(active_cooldowns) - time.time()
                    raise RuntimeError(
                        "All API keys are rate-limited or over budget. "
                        "Shortest cooldown: {:.0f}s".format(max(0, wait_s))
                    )
                raise RuntimeError(
                    "No API keys are available. Check budgets and key validity."
                )
            return max(available, key=lambda k: k.health_score())

    def report_usage(self, alias: str, tokens_in: int, tokens_out: int) -> None:
        with self._lock:
            for k in self._keys:
                if k.alias == alias:
                    k.record_usage(tokens_in, tokens_out)
                    k.record_success()
                    break

    def report_error(self, alias: str, status: int) -> None:
        with self._lock:
            for k in self._keys:
                if k.alias == alias:
                    k.record_error(status)
                    break

    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(k.estimated_cost_usd() for k in self._keys)

    def summary(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "alias": k.alias,
                    "available": k.is_available(),
                    "health": round(k.health_score(), 3),
                    "cost_usd": round(k.estimated_cost_usd(), 5),
                    "budget_remaining_usd": round(k.budget_remaining(), 5),
                    "tokens_in": k.tokens_in,
                    "tokens_out": k.tokens_out,
                    "errors_429": k.errors_429,
                }
                for k in self._keys
            ]


_manager: Optional[KeyManager] = None
_manager_lock = threading.Lock()


def get_key_manager() -> KeyManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = KeyManager()
    return _manager
