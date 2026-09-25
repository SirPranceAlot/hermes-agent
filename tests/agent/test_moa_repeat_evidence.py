"""MoA loop claims require evidence from the full, untruncated transcript."""

import json

from agent.moa_loop import _REFERENCE_SYSTEM_PROMPT, _repeat_evidence


def _call(name, args, call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def _result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _round(name, args, call_id, result):
    return [_call(name, args, call_id), _result(call_id, result)]


def _repeated_read(count=3):
    messages = [{"role": "user", "content": "check the page"}]
    for index in range(count):
        messages.extend(_round("read_file", {"path": "page.md"}, f"c{index}", "same page"))
    return messages


def test_two_identical_reads_are_not_loop_evidence():
    assert _repeat_evidence(_repeated_read(2)) == ""


def test_three_identical_reads_with_same_result_are_reported_with_call_indices():
    evidence = _repeat_evidence(_repeated_read(3))

    assert "read_file" in evidence
    assert "page.md" in evidence
    assert "messages 2, 4, 6" in evidence


def test_a_write_breaks_a_repeated_read_streak():
    messages = [{"role": "user", "content": "check the page"}]
    messages.extend(_round("read_file", {"path": "page.md"}, "c1", "same page"))
    messages.extend(_round("read_file", {"path": "page.md"}, "c2", "same page"))
    messages.extend(_round("write_file", {"path": "page.md", "content": "new"}, "c3", "written"))
    messages.extend(_round("read_file", {"path": "page.md"}, "c4", "same page"))

    assert _repeat_evidence(messages) == ""


def test_skill_reload_after_pruned_result_is_not_loop_evidence():
    messages = [{"role": "user", "content": "load this skill"}]
    args = {"name": "simple-english"}
    messages.extend(_round("skill_view", args, "c1", "[SKILL_PRUNED: reload this skill]"))
    messages.extend(_round("skill_view", args, "c2", "full skill contents"))
    messages.extend(_round("skill_view", args, "c3", "full skill contents"))

    assert _repeat_evidence(messages) == ""


def test_retry_after_error_breaks_an_existing_repeat_streak():
    messages = [{"role": "user", "content": "fetch the page"}]
    args = {"id": "page-1"}
    name = "mcp__notion__notion__notion_fetch"
    messages.extend(_round(name, args, "c1", "page contents"))
    messages.extend(_round(name, args, "c2", "page contents"))
    messages.extend(_round(name, args, "c3", '{"error":"temporary timeout"}'))
    messages.extend(_round(name, args, "c4", "page contents"))
    messages.extend(_round(name, args, "c5", "page contents"))

    assert _repeat_evidence(messages) == ""


def test_advisor_evidence_is_also_in_aggregator_guidance():
    from agent.moa_loop import MoAChatCompletions, _RefAccounting

    facade = MoAChatCompletions.__new__(MoAChatCompletions)
    facade.preset_name = "review"
    facade._privacy_mode = ""
    evidence = "[Hermes repeat evidence] read_file page.md repeated at messages 2, 4, 6."
    guidance = facade._build_guidance(
        [("advisor", "STATUS: looping", _RefAccounting(None))],
        {"provider": "openrouter", "model": "test"},
        "loud",
        repeat_evidence=evidence,
    )

    assert guidance is not None
    assert evidence in guidance
    assert "Hermes repeat evidence lists the same tool and arguments" in guidance


def test_process_polling_is_not_loop_evidence():
    messages = [{"role": "user", "content": "wait for the process"}]
    for index in range(3):
        messages.extend(_round("process_manage", {"action": "poll", "session_id": "p1"}, f"c{index}", "still running"))

    assert _repeat_evidence(messages) == ""


def test_wrapped_notion_fetch_is_unwrapped_and_reported():
    messages = [{"role": "user", "content": "check the page"}]
    wrapped = {"calls": [{"name": "mcp__notion__notion__notion_fetch", "arguments": {"id": "page-1"}}]}
    for index in range(3):
        messages.extend(_round("tool_call", wrapped, f"c{index}", "<untrusted_tool_result>same page</untrusted_tool_result>"))

    evidence = _repeat_evidence(messages)

    assert "mcp__notion__notion__notion_fetch" in evidence
    assert "page-1" in evidence
    assert "tool_call" not in evidence


def test_repeat_evidence_redacts_and_caps_argument_preview():
    args = {"contact": "person@example.com", "padding": "x" * 500}
    messages = [{"role": "user", "content": "check the page"}]
    for index in range(3):
        messages.extend(_round("read_file", args, f"c{index}", "same result"))

    evidence = _repeat_evidence(messages)

    assert "person@example.com" not in evidence
    assert "[redacted email]" in evidence
    assert len(evidence) < 400


def test_repeat_evidence_is_deterministic():
    messages = _repeated_read(3)

    assert _repeat_evidence(messages) == _repeat_evidence(messages)


def test_advisor_must_not_claim_a_loop_without_listed_evidence():
    assert "only when the Hermes repeat evidence block lists the call" in _REFERENCE_SYSTEM_PROMPT
    assert "possible repeat" in _REFERENCE_SYSTEM_PROMPT
