"""The job of this file is to manage the key pool, decide which key is safe to use next, and 
remember what the last Groq headers said."""


from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Optional, TypedDict


UTC = timezone.utc
RESET_WINDOW_PATTERN = re.compile(
    r"^(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+(?:\.\d+)?)s)?$"
)


class KeyConfig(TypedDict):
    label: str
    provider: str
    api_key: str
    enabled: bool


class NoEligibleKeyError(RuntimeError):
    def __init__(self, message: str, retry_at: Optional[datetime] = None) -> None:
        super().__init__(message)
        self.retry_at = retry_at


@dataclass(slots=True)
class KeyState:
    label: str
    provider: str
    api_key: str
    enabled: bool = True

    remaining_tokens: Optional[int] = None
    remaining_requests: Optional[int] = None

    reset_tokens_at: Optional[datetime] = None
    reset_requests_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None

    last_used_at: Optional[datetime] = None
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    disabled_reason: Optional[str] = None

    def refresh(self, now: Optional[datetime] = None) -> None:
        now = now or utcnow()

        if self.cooldown_until and now >= self.cooldown_until:
            self.cooldown_until = None

        if self.reset_tokens_at and now >= self.reset_tokens_at:
            self.remaining_tokens = None
            self.reset_tokens_at = None

        if self.reset_requests_at and now >= self.reset_requests_at:
            self.remaining_requests = None
            self.reset_requests_at = None

    @property
    def is_disabled(self) -> bool:
        return (not self.enabled) or (self.disabled_reason is not None)

    def is_in_cooldown(self, now: Optional[datetime] = None) -> bool:
        now = now or utcnow()
        self.refresh(now)
        return self.cooldown_until is not None and now < self.cooldown_until

    def projected_headroom(self, request_cost_tokens: int) -> Optional[int]:
        if self.remaining_tokens is None:
            return None
        return self.remaining_tokens - request_cost_tokens

    def has_budget_for(
        self,
        request_cost_tokens: int,
        reserve_tokens: int = 0,
        now: Optional[datetime] = None,
    ) -> bool:
        now = now or utcnow()
        self.refresh(now)

        if self.is_disabled or self.is_in_cooldown(now):
            return False

        if self.remaining_requests is not None and self.remaining_requests < 1:
            return False

        if self.remaining_tokens is not None:
            if (self.remaining_tokens - request_cost_tokens) < reserve_tokens:
                return False

        return True

    def as_public_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "provider": self.provider,
            "enabled": self.enabled,
            "remaining_tokens": self.remaining_tokens,
            "remaining_requests": self.remaining_requests,
            "reset_tokens_at": iso_or_none(self.reset_tokens_at),
            "reset_requests_at": iso_or_none(self.reset_requests_at),
            "cooldown_until": iso_or_none(self.cooldown_until),
            "last_used_at": iso_or_none(self.last_used_at),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "disabled_reason": self.disabled_reason,
        }


class KeyMaster:
    def __init__(
        self,
        keys_path: str | Path = "keys.json",
        *,
        reserve_tokens: int = 300,
        transient_cooldown_seconds: int = 5,
    ) -> None:
        self.keys_path = Path(keys_path)
        self.reserve_tokens = reserve_tokens
        self.transient_cooldown_seconds = transient_cooldown_seconds
        self._states: dict[str, KeyState] = {}
        self.reload()

    def reload(self) -> None:
        self._states = self._load_states(self.keys_path)

    def all_states(self) -> list[KeyState]:
        now = utcnow()
        for state in self._states.values():
            state.refresh(now)
        return list(self._states.values())

    def snapshot(self) -> list[dict[str, object]]:
        return [state.as_public_dict() for state in self.all_states()]

    def get_best_key(self, request_cost_tokens: int) -> KeyState:
        now = utcnow()

        candidates = [
            state
            for state in self._states.values()
            if state.has_budget_for(
                request_cost_tokens=request_cost_tokens,
                reserve_tokens=self.reserve_tokens,
                now=now,
            )
        ]

        if not candidates:
            retry_at = self.next_available_at(now)
            raise NoEligibleKeyError(
                "No eligible Groq key is currently available for this request.",
                retry_at=retry_at,
            )

        candidates.sort(key=lambda state: self._selection_rank(state, request_cost_tokens))

        chosen = candidates[0]
        chosen.last_used_at = now
        return chosen

    def mark_success(
        self,
        label: str,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        state = self._require_state(label)
        now = utcnow()

        state.last_used_at = now
        state.consecutive_failures = 0
        state.last_error = None
        state.cooldown_until = None

        if not headers:
            return

        remaining_tokens = header_int(headers, "x-ratelimit-remaining-tokens")
        remaining_requests = header_int(headers, "x-ratelimit-remaining-requests")
        reset_tokens = header_reset_at(headers, "x-ratelimit-reset-tokens", now)
        reset_requests = header_reset_at(headers, "x-ratelimit-reset-requests", now)

        if remaining_tokens is not None:
            state.remaining_tokens = remaining_tokens
        if remaining_requests is not None:
            state.remaining_requests = remaining_requests
        if reset_tokens is not None:
            state.reset_tokens_at = reset_tokens
        if reset_requests is not None:
            state.reset_requests_at = reset_requests

    def mark_rate_limited(
        self,
        label: str,
        *,
        retry_after: Optional[str | int | float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        state = self._require_state(label)
        now = utcnow()

        state.last_used_at = now
        state.consecutive_failures += 1
        state.last_error = error_message or "rate_limited"

        cooldown_seconds = coerce_seconds(retry_after) or self.transient_cooldown_seconds
        state.cooldown_until = now + timedelta(seconds=cooldown_seconds)

    def mark_transient_failure(
        self,
        label: str,
        *,
        error_message: str,
    ) -> None:
        state = self._require_state(label)
        now = utcnow()

        state.last_used_at = now
        state.consecutive_failures += 1
        state.last_error = error_message

        backoff_seconds = min(
            60,
            self.transient_cooldown_seconds * (2 ** max(0, state.consecutive_failures - 1)),
        )
        state.cooldown_until = now + timedelta(seconds=backoff_seconds)

    def mark_auth_failure(self, label: str, *, error_message: str) -> None:
        state = self._require_state(label)
        now = utcnow()

        state.last_used_at = now
        state.last_error = error_message
        state.disabled_reason = error_message
        state.enabled = False
        state.cooldown_until = None

    def mark_oversized_request(self, label: str, *, error_message: str) -> None:
        state = self._require_state(label)
        state.last_error = error_message
        # Deliberately no cooldown or disable here.
        # An oversized request is a request-shape problem, not a key-health problem.

    def next_available_at(self, now: Optional[datetime] = None) -> Optional[datetime]:
        now = now or utcnow()
        times: list[datetime] = []

        for state in self._states.values():
            state.refresh(now)

            if state.is_disabled:
                continue

            if state.cooldown_until and state.cooldown_until > now:
                times.append(state.cooldown_until)

            if state.reset_tokens_at and state.reset_tokens_at > now:
                times.append(state.reset_tokens_at)

            if state.reset_requests_at and state.reset_requests_at > now:
                times.append(state.reset_requests_at)

        return min(times) if times else None

    def _require_state(self, label: str) -> KeyState:
        try:
            return self._states[label]
        except KeyError as exc:
            raise KeyError(f"Unknown key label: {label}") from exc

    def _selection_rank(self, state: KeyState, request_cost_tokens: int) -> tuple:
        unseen = (
            state.last_used_at is None
            and state.remaining_tokens is None
            and state.remaining_requests is None
        )

        if unseen:
            freshness_bucket = 0
        elif state.remaining_tokens is not None:
            freshness_bucket = 1
        else:
            freshness_bucket = 2

        headroom = state.projected_headroom(request_cost_tokens)
        headroom_rank = -(headroom if headroom is not None else 0)

        last_used_rank = (
            state.last_used_at.timestamp() if state.last_used_at is not None else float("-inf")
        )

        return (
            freshness_bucket,
            state.consecutive_failures,
            headroom_rank,
            last_used_rank,
            state.label,
        )

    def _load_states(self, keys_path: Path) -> dict[str, KeyState]:
        if not keys_path.exists():
            raise FileNotFoundError(f"Keys file not found: {keys_path}")

        with keys_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError("keys.json must contain a top-level JSON object.")

        raw_keys = payload.get("keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise ValueError("keys.json must contain a non-empty 'keys' array.")

        states: dict[str, KeyState] = {}

        for index, raw in enumerate(raw_keys):
            if not isinstance(raw, dict):
                raise ValueError(f"Key entry at index {index} must be an object.")

            config = validate_key_config(raw, index=index)

            if config["label"] in states:
                raise ValueError(f"Duplicate key label found: {config['label']}")

            states[config["label"]] = KeyState(
                label=config["label"],
                provider=config["provider"],
                api_key=config["api_key"],
                enabled=config["enabled"],
            )

        return states


def validate_key_config(raw: Mapping[str, object], *, index: int) -> KeyConfig:
    api_key = raw.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"Key entry at index {index} is missing a valid 'api_key'.")

    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        label = f"groq-key-{index + 1}"

    provider = raw.get("provider", "groq")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"Key entry at index {index} has an invalid 'provider'.")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"Key entry at index {index} has an invalid 'enabled' flag.")

    return {
        "label": label.strip(),
        "provider": provider.strip(),
        "api_key": api_key.strip(),
        "enabled": enabled,
    }


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def iso_or_none(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def header_value(headers: Mapping[str, str], name: str) -> Optional[str]:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def header_int(headers: Mapping[str, str], name: str) -> Optional[int]:
    raw = header_value(headers, name)
    if raw is None:
        return None

    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def header_reset_at(
    headers: Mapping[str, str],
    name: str,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    raw = header_value(headers, name)
    if raw is None:
        return None

    seconds = parse_reset_window_seconds(raw)
    if seconds is None:
        return None

    now = now or utcnow()
    return now + timedelta(seconds=seconds)


def parse_reset_window_seconds(raw: str) -> Optional[float]:
    match = RESET_WINDOW_PATTERN.match(raw.strip())
    if not match:
        return None

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0.0)

    total_seconds = (hours * 3600) + (minutes * 60) + seconds
    return total_seconds if total_seconds >= 0 else None


def coerce_seconds(value: Optional[str | int | float]) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return None
