import json
from pathlib import Path
from typing import Any, Optional

import pytest

from ptiq.core.groq_client import PTIQGroqClient, TaskProfile, resolve_groq_base_url
from ptiq.core.keymaster import KeyMaster


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        json_body: Optional[dict[str, Any]] = None,
        text: str = "",
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self._json_body = json_body
        self.text = text
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict[str, Any]:
        if self._json_body is None:
            raise ValueError("No JSON body")
        return self._json_body


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses[:]
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )

        if not self._responses:
            raise AssertionError("FakeSession ran out of queued responses.")

        return self._responses.pop(0)

    def close(self) -> None:
        return None


def write_keys_file(tmp_path: Path, key_count: int) -> Path:
    keys = []
    for index in range(1, key_count + 1):
        keys.append(
            {
                "label": f"k{index}",
                "provider": "groq",
                "api_key": f"fake-key-{index}",
                "enabled": True,
            }
        )

    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": keys}), encoding="utf-8")
    return path


def build_client(keys_path: Path, session: FakeSession) -> tuple[PTIQGroqClient, KeyMaster]:
    keymaster = KeyMaster(keys_path=keys_path, reserve_tokens=100)

    profiles = {
        "test_task": TaskProfile(
            name="test_task",
            model="llama-3.1-8b-instant",
            max_output_tokens=1000,
            min_output_tokens=200,
            temperature=0.1,
            timeout_seconds=30,
            json_mode=False,
        )
    }

    client = PTIQGroqClient(
        keymaster=keymaster,
        task_profiles=profiles,
        session=session,
    )
    return client, keymaster


def success_payload(content: str = "ok") -> dict[str, Any]:
    return {
        "model": "llama-3.1-8b-instant",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200,
        },
    }


def test_oversized_error_shrinks_output_budget_before_giving_up(tmp_path: Path) -> None:
    keys_path = write_keys_file(tmp_path, key_count=1)

    session = FakeSession(
        responses=[
            FakeResponse(
                status_code=400,
                json_body={
                    "error": {
                        "message": "Requested too many tokens: limit 12000, requested 13000."
                    }
                },
            ),
            FakeResponse(
                status_code=200,
                json_body=success_payload("shrunk request succeeded"),
                headers={
                    "x-ratelimit-remaining-tokens": "9000",
                    "x-ratelimit-remaining-requests": "29",
                    "x-ratelimit-reset-tokens": "30s",
                    "x-ratelimit-reset-requests": "30s",
                },
            ),
        ]
    )

    client, keymaster = build_client(keys_path, session)

    response = client.generate_text(
        "test_task",
        [{"role": "user", "content": "Analyze this tender."}],
    )

    assert response.text == "shrunk request succeeded"
    assert len(session.calls) == 2
    assert session.calls[0]["json"]["max_tokens"] == 1000
    assert session.calls[1]["json"]["max_tokens"] == 700

    state = keymaster.all_states()[0]
    assert state.cooldown_until is None
    assert state.disabled_reason is None
    assert state.consecutive_failures == 0


def test_rate_limit_cools_first_key_and_fails_over_to_second(tmp_path: Path) -> None:
    keys_path = write_keys_file(tmp_path, key_count=2)

    session = FakeSession(
        responses=[
            FakeResponse(
                status_code=429,
                json_body={"error": {"message": "Rate limit exceeded."}},
                headers={"retry-after": "5"},
            ),
            FakeResponse(
                status_code=200,
                json_body=success_payload("second key worked"),
                headers={
                    "x-ratelimit-remaining-tokens": "8500",
                    "x-ratelimit-remaining-requests": "25",
                    "x-ratelimit-reset-tokens": "20s",
                    "x-ratelimit-reset-requests": "20s",
                },
            ),
        ]
    )

    client, keymaster = build_client(keys_path, session)

    response = client.generate_text(
        "test_task",
        [{"role": "user", "content": "Analyze this tender."}],
    )

    assert response.text == "second key worked"
    assert response.key_label == "k2"
    assert len(session.calls) == 2

    assert session.calls[0]["headers"]["Authorization"] == "Bearer fake-key-1"
    assert session.calls[1]["headers"]["Authorization"] == "Bearer fake-key-2"

    states = {state.label: state for state in keymaster.all_states()}
    assert states["k1"].cooldown_until is not None
    assert states["k1"].disabled_reason is None
    assert states["k2"].disabled_reason is None


def test_success_updates_live_rate_limit_state_from_headers(tmp_path: Path) -> None:
    keys_path = write_keys_file(tmp_path, key_count=1)

    session = FakeSession(
        responses=[
            FakeResponse(
                status_code=200,
                json_body=success_payload("header sync success"),
                headers={
                    "x-ratelimit-remaining-tokens": "7777",
                    "x-ratelimit-remaining-requests": "17",
                    "x-ratelimit-reset-tokens": "45s",
                    "x-ratelimit-reset-requests": "1m30s",
                },
            )
        ]
    )

    client, keymaster = build_client(keys_path, session)

    response = client.generate_text(
        "test_task",
        [{"role": "user", "content": "Analyze this tender."}],
    )

    state = keymaster.all_states()[0]

    assert response.text == "header sync success"
    assert state.remaining_tokens == 7777
    assert state.remaining_requests == 17
    assert state.reset_tokens_at is not None
    assert state.reset_requests_at is not None


def test_resolve_groq_base_url_rejects_unapproved_host_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_BASE_URL", "https://evil.example.com/openai/v1")
    monkeypatch.delenv("GROQ_ALLOW_CUSTOM_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="unapproved host"):
        resolve_groq_base_url()
