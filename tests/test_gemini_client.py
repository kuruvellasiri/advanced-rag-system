from types import ModuleType
import sys

from src.rag import gemini_client


def test_fixed_sampling_model_is_initialized_without_temperature(monkeypatch):
    captured = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = ModuleType("langchain_google_genai")
    fake_module.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    monkeypatch.setattr(gemini_client, "ensure_api_key", lambda: None)

    gemini_client.create_chat_client("gemini-3.5-flash-lite", temperature=0.0)

    assert "temperature" not in captured
    assert captured["model"] == "gemini-3.5-flash-lite"
    assert captured["retries"] == 0


def test_configurable_model_keeps_requested_temperature(monkeypatch):
    captured = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = ModuleType("langchain_google_genai")
    fake_module.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    monkeypatch.setattr(gemini_client, "ensure_api_key", lambda: None)

    gemini_client.create_chat_client("gemini-2.5-flash", temperature=0.2)

    assert captured["temperature"] == 0.2
