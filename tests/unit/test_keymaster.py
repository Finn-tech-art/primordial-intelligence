from ptiq.core.keymaster import KeyState


def test_keystate_repr_does_not_expose_api_key() -> None:
    state = KeyState(
        label="k1",
        provider="groq",
        api_key="super-secret-key",
    )

    rendered = repr(state)

    assert "super-secret-key" not in rendered
    assert "api_key" not in rendered
