"""Single Groq entry point for all PT IQ agents.

This client handles:
- token preflight estimation
- key selection through KeyMaster
- retries and cooldown-aware failover
- oversized-request detection
- response normalization for text and JSON tasks
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import requests

from .keymaster import KeyMaster, NoEligibleKeyError


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OVERSIZED_ERROR_SNIPPETS = (
    "requested too many tokens",
    "requested tokens",
    "context length",
    "maximum context length",
    "max tokens",
    "please reduce the length",
    "prompt is too long",
)


@dataclass(slots=True, frozen=True)
class TaskProfile:
    name: str
    model: str
    max_output_tokens: int
    min_output_tokens: int
    temperature: float
    timeout_seconds: int
    json_mode: bool = False
    allow_output_shrink: bool = True
    max_attempts: int = 8


@dataclass(slots=True)
class GroqResponse:
    task_name: str
    model: str
    key_label: str
    text: str
    json_data: Optional[dict[str, Any]]
    usage: dict[str, Any]
    headers: dict[str, str]


class GroqClientError(RuntimeError):
    """Base error for PT IQ Groq client failures."""


class GroqRateLimitError(GroqClientError):
    def __init__(self, message: str, retry_at: Optional[datetime] = None) -> None:
        super().__init__(message)
        self.retry_at = retry_at


class NeedsChunkingError(GroqClientError):
    """Raised when a request is too large and should be chunked or compressed."""


class JSONResponseError(GroqClientError):
    """Raised when a JSON-mode task returns non-JSON content."""


class PTIQGroqClient:
    """Operational wrapper around Groq's chat completions API."""

    def __init__(
        self,
        *,
        keymaster: KeyMaster,
        task_profiles: Mapping[str, TaskProfile],
        base_url: str = GROQ_BASE_URL,
        default_timeout_seconds: int = 60,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.keymaster = keymaster
        self.task_profiles = dict(task_profiles)
        self.base_url = base_url.rstrip("/")
        self.default_timeout_seconds = default_timeout_seconds
        self.session = session or requests.Session()

    def close(self) -> None:
        self.session.close()

    def generate_json(
        self,
        task_name: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GroqResponse:
        response = self.request(
            task_name,
            messages,
            expect_json=True,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        if response.json_data is None:
            raise JSONResponseError(
                f"Task '{task_name}' expected JSON, but the model returned non-JSON content."
            )
        return response

    def generate_text(
        self,
        task_name: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GroqResponse:
        return self.request(
            task_name,
            messages,
            expect_json=False,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    def request(
        self,
        task_name: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        expect_json: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GroqResponse:
        if not messages:
            raise ValueError("Groq request requires at least one message.")

        profile = self._require_task_profile(task_name)
        expect_json = profile.json_mode if expect_json is None else expect_json

        current_output_tokens = (
            profile.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        current_temperature = profile.temperature if temperature is None else temperature

        if current_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0.")

        attempt_limit = max(profile.max_attempts, max(1, len(self.keymaster.all_states())) * 2)
        last_error: Optional[Exception] = None

        for _ in range(attempt_limit):
            estimated_cost = self.estimate_request_tokens(
                messages=messages,
                max_output_tokens=current_output_tokens,
            )

            try:
                key_state = self.keymaster.get_best_key(estimated_cost)
            except NoEligibleKeyError as exc:
                shrunk = self._maybe_shrink_output_budget(current_output_tokens, profile)
                if shrunk is not None:
                    current_output_tokens = shrunk
                    continue

                if exc.retry_at is not None:
                    raise GroqRateLimitError(
                        "No eligible Groq key is currently available for this request.",
                        retry_at=exc.retry_at,
                    ) from exc

                raise GroqClientError(
                    "No eligible Groq key is available. Keys may be disabled or exhausted."
                ) from exc

            payload = self._build_payload(
                profile=profile,
                messages=messages,
                max_output_tokens=current_output_tokens,
                temperature=current_temperature,
                expect_json=expect_json,
            )

            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(key_state.api_key),
                    json=payload,
                    timeout=(
                        profile.timeout_seconds
                        if profile.timeout_seconds > 0
                        else self.default_timeout_seconds
                    ),
                )
            except requests.RequestException as exc:
                self.keymaster.mark_transient_failure(
                    key_state.label,
                    error_message=f"network_error: {exc}",
                )
                last_error = exc
                continue

            if response.ok:
                parsed = self._safe_json(response)
                text = self._extract_content(parsed)
                json_data = self._parse_json_text(text) if expect_json else None
                usage = self._extract_usage(parsed)

                self.keymaster.mark_success(key_state.label, headers=response.headers)

                return GroqResponse(
                    task_name=task_name,
                    model=parsed.get("model", profile.model),
                    key_label=key_state.label,
                    text=text,
                    json_data=json_data,
                    usage=usage,
                    headers=dict(response.headers),
                )

            error_payload = self._safe_json(response)
            error_message = self._extract_error_message(error_payload) or response.text.strip()
            error_message = error_message or f"Groq request failed with HTTP {response.status_code}"

            if self._looks_oversized(response.status_code, error_message):
                self.keymaster.mark_oversized_request(
                    key_state.label,
                    error_message=error_message,
                )

                shrunk = self._maybe_shrink_output_budget(current_output_tokens, profile)
                if shrunk is not None:
                    current_output_tokens = shrunk
                    last_error = NeedsChunkingError(error_message)
                    continue

                raise NeedsChunkingError(
                    "Groq rejected the request as too large. "
                    "Chunk or compress the input before retrying."
                )

            if response.status_code == 429:
                self.keymaster.mark_rate_limited(
                    key_state.label,
                    retry_after=response.headers.get("retry-after"),
                    error_message=error_message,
                )
                last_error = GroqRateLimitError(
                    error_message,
                    retry_at=self.keymaster.next_available_at(),
                )
                continue

            if response.status_code in (401, 403):
                self.keymaster.mark_auth_failure(
                    key_state.label,
                    error_message=error_message,
                )
                last_error = GroqClientError(error_message)
                continue

            if response.status_code >= 500:
                self.keymaster.mark_transient_failure(
                    key_state.label,
                    error_message=error_message,
                )
                last_error = GroqClientError(error_message)
                continue

            raise GroqClientError(error_message)

        if isinstance(last_error, GroqRateLimitError):
            raise last_error

        if isinstance(last_error, NeedsChunkingError):
            raise last_error

        if isinstance(last_error, Exception):
            raise GroqClientError(f"Groq request failed after retries: {last_error}") from last_error

        raise GroqClientError("Groq request failed after retries for an unknown reason.")

    def estimate_request_tokens(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        max_output_tokens: int,
    ) -> int:
        """Conservative token estimate for admission control.

        Important:
        - this method estimates the raw request size
        - reserve headroom is enforced by KeyMaster, not here
        """
        message_overhead = 0
        content_tokens = 0

        for message in messages:
            message_overhead += 6
            content_tokens += self._estimate_content_tokens(message)

        return content_tokens + message_overhead + max_output_tokens

    def _require_task_profile(self, task_name: str) -> TaskProfile:
        try:
            return self.task_profiles[task_name]
        except KeyError as exc:
            known = ", ".join(sorted(self.task_profiles))
            raise KeyError(f"Unknown task profile '{task_name}'. Known tasks: {known}") from exc

    def _build_payload(
        self,
        *,
        profile: TaskProfile,
        messages: Sequence[Mapping[str, Any]],
        max_output_tokens: int,
        temperature: float,
        expect_json: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": profile.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

        if expect_json:
            payload["response_format"] = {"type": "json_object"}

        return payload

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _estimate_content_tokens(self, payload: Any) -> int:
        serialized = self._serialize_for_estimation(payload)
        if not serialized:
            return 0
        return math.ceil(len(serialized) / 4)

    def _serialize_for_estimation(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)

        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return str(value)

    def _safe_json(self, response: requests.Response) -> dict[str, Any]:
        try:
            parsed = response.json()
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _extract_content(self, payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first = choices[0]
        if not isinstance(first, dict):
            return ""

        message = first.get("message")
        if not isinstance(message, dict):
            return ""

        content = message.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )

        return str(content or "")

    def _extract_usage(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        usage = payload.get("usage")
        return usage if isinstance(usage, dict) else {}

    def _extract_error_message(self, payload: Mapping[str, Any]) -> str:
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message.strip()
        return ""

    def _parse_json_text(self, text: str) -> Optional[dict[str, Any]]:
        cleaned = text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```JSON"):
            cleaned = cleaned[len("```JSON"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        if not cleaned:
            return None

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    def _looks_oversized(self, status_code: int, message: str) -> bool:
        if status_code == 413:
            return True

        lowered = message.lower()
        return any(snippet in lowered for snippet in OVERSIZED_ERROR_SNIPPETS)

    def _maybe_shrink_output_budget(
        self,
        current_output_tokens: int,
        profile: TaskProfile,
    ) -> Optional[int]:
        if not profile.allow_output_shrink:
            return None

        if current_output_tokens <= profile.min_output_tokens:
            return None

        shrunk = max(profile.min_output_tokens, math.floor(current_output_tokens * 0.7))
        return shrunk if shrunk < current_output_tokens else None


def default_task_profiles() -> dict[str, TaskProfile]:
    extraction_model = os.getenv("GROQ_EXTRACTION_MODEL", "llama-3.1-8b-instant")
    reasoning_model = os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")
    min_output_tokens = int(os.getenv("GROQ_MIN_OUTPUT_TOKENS", "300"))
    default_timeout = int(os.getenv("GROQ_REQUEST_TIMEOUT_SECONDS", "60"))

    return {
        "business_dna_extraction": TaskProfile(
            name="business_dna_extraction",
            model=extraction_model,
            max_output_tokens=700,
            min_output_tokens=min_output_tokens,
            temperature=0.1,
            timeout_seconds=default_timeout,
            json_mode=True,
        ),
        "ibp_extraction": TaskProfile(
            name="ibp_extraction",
            model=extraction_model,
            max_output_tokens=900,
            min_output_tokens=min_output_tokens,
            temperature=0.1,
            timeout_seconds=default_timeout,
            json_mode=True,
        ),
        "deep_match": TaskProfile(
            name="deep_match",
            model=reasoning_model,
            max_output_tokens=1400,
            min_output_tokens=600,
            temperature=0.2,
            timeout_seconds=max(default_timeout, 90),
            json_mode=True,
        ),
    }


def build_default_client() -> PTIQGroqClient:
    keys_path = os.getenv("GROQ_KEYS_PATH", "keys.json")
    reserve_tokens = int(os.getenv("GROQ_TOKEN_SAFETY_MARGIN", "500"))
    timeout_seconds = int(os.getenv("GROQ_REQUEST_TIMEOUT_SECONDS", "60"))

    keymaster = KeyMaster(
        keys_path=keys_path,
        reserve_tokens=reserve_tokens,
    )

    return PTIQGroqClient(
        keymaster=keymaster,
        task_profiles=default_task_profiles(),
        base_url=os.getenv("GROQ_BASE_URL", GROQ_BASE_URL),
        default_timeout_seconds=timeout_seconds,
    )


_default_client: Optional[PTIQGroqClient] = None


def get_groq_client() -> PTIQGroqClient:
    global _default_client

    if _default_client is None:
        _default_client = build_default_client()

    return _default_client
