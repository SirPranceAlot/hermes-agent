"""MoA aggregator: a destination that rejects the reasoning field gets it omitted.

The streaming aggregator path returns before ``call_llm``'s parameter ladder, and the main loop's
``_reasoning_effort_rejected`` flag never reaches the aggregator call, so without this recovery a
LiteLLM ``UnsupportedParamsError: ... ['reasoning_effort']`` 400 repeats on every retry.
"""

from types import SimpleNamespace

import pytest

from agent import moa_loop

LITELLM_MSG = (
    "Error code: 400 - {'error': {'message': \"litellm.UnsupportedParamsError: bedrock_mantle does not "
    "support parameters: ['reasoning_effort'], for model=openai.gpt-6-luna. To drop these, set "
    "`litellm.drop_params=True`\", 'type': 'None', 'param': None, 'code': '400'}}"
)


class _ReasoningRejected(Exception):
    status_code = 400


def _destination(calls, rejecting_models):
    def call_llm(**kw):
        calls.append(kw)
        if kw["model"] in rejecting_models and kw.get("reasoning_config") is not None:
            raise _ReasoningRejected(LITELLM_MSG)
        return SimpleNamespace(choices=[])
    return call_llm


@pytest.fixture
def facade(monkeypatch):
    monkeypatch.setattr(
        moa_loop, "_slot_runtime",
        lambda slot: {"provider": "custom", "model": slot["model"], "base_url": "http://relay.local/v1",
                      "api_mode": "chat_completions"},
    )
    monkeypatch.setattr(moa_loop, "_aggregator_reasoning_config", lambda agg: {"enabled": True, "effort": "medium"})
    f = moa_loop.MoAChatCompletions("default", agent=None)
    f._pending_trace = None
    return f


def _send(facade, model):
    prepared = facade.rebase_prepared_request(
        {"guidance": "advice", "aggregator": {"provider": "custom", "model": model}, "aggregator_temperature": None},
        [{"role": "user", "content": "task"}],
    )
    return facade._call_prepared_aggregator(prepared, {"tools": None, "stream": True})


def test_rejected_reasoning_field_is_dropped_once_and_remembered(monkeypatch, facade):
    calls = []
    monkeypatch.setattr(moa_loop, "call_llm", _destination(calls, {"strict"}))

    _send(facade, "strict")
    assert [c["reasoning_config"] for c in calls] == [{"enabled": True, "effort": "medium"}, None]

    del calls[:]
    _send(facade, "strict")  # remembered: no second 400
    assert [c["reasoning_config"] for c in calls] == [None]


def test_accepting_destination_keeps_its_reasoning_config(monkeypatch, facade):
    calls = []
    monkeypatch.setattr(moa_loop, "call_llm", _destination(calls, {"strict"}))

    _send(facade, "strict")
    del calls[:]
    _send(facade, "lenient")
    assert [c["reasoning_config"] for c in calls] == [{"enabled": True, "effort": "medium"}]
