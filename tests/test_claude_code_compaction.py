import json
from pathlib import Path

from pier.agents.installed.claude_code import ClaudeCode, _is_compaction_boundary


def test_top_level_compact_boundary_is_detected():
    # Real Claude Code session artifacts emit the marker as a top-level subtype.
    event = {
        "type": "system",
        "subtype": "compact_boundary",
        "compactMetadata": {"trigger": "auto"},
    }
    assert _is_compaction_boundary(event) is True


def test_non_compaction_system_events_are_ignored():
    assert _is_compaction_boundary({"type": "system", "subtype": "init"}) is False
    assert _is_compaction_boundary({"type": "system"}) is False
    # Wrong event type, even with a matching subtype, is not a boundary.
    assert (
        _is_compaction_boundary({"type": "user", "subtype": "compact_boundary"})
        is False
    )


def test_final_result_metrics_use_last_aggregate_result(tmp_path: Path):
    agent = ClaudeCode(logs_dir=tmp_path, model_name="claude-sonnet")
    results = [
        {
            "type": "result",
            "total_cost_usd": 0.1,
            "modelUsage": {
                "claude-sonnet": {
                    "inputTokens": 1,
                    "cacheReadInputTokens": 2,
                    "cacheCreationInputTokens": 3,
                    "outputTokens": 4,
                    "costUSD": 0.1,
                }
            },
        },
        {
            "type": "result",
            "total_cost_usd": 0.9,
            "modelUsage": {
                "claude-sonnet": {
                    "inputTokens": 10,
                    "cacheReadInputTokens": 20,
                    "cacheCreationInputTokens": 30,
                    "outputTokens": 40,
                    "costUSD": 0.9,
                }
            },
        },
    ]
    (tmp_path / "claude-code.txt").write_text(
        "\n".join(json.dumps(result) for result in results), encoding="utf-8"
    )

    metrics = agent._parse_final_result_metrics_from_stream_json()

    assert metrics == {
        "total_prompt_tokens": 60,
        "total_completion_tokens": 40,
        "total_cached_tokens": 20,
        "total_cache_creation_input_tokens": 30,
        "total_cache_read_input_tokens": 20,
        "total_cost_usd": 0.9,
    }


def test_subagent_jsonl_is_saved_and_referenced(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    session_dir = logs_dir / "sessions" / "projects" / "-app"
    subagents_dir = session_dir / "parent-session" / "subagents"
    subagents_dir.mkdir(parents=True)

    parent_events = [
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:00Z",
            "sessionId": "parent-session",
            "version": "2.1.187",
            "message": {
                "id": "parent-message",
                "role": "assistant",
                "model": "claude-sonnet",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-agent-1",
                        "name": "Agent",
                        "input": {"description": "Explore code"},
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:01Z",
            "sessionId": "parent-session",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-agent-1",
                        "content": "Agent launched",
                    }
                ],
            },
        },
    ]
    (session_dir / "parent-session.jsonl").write_text(
        "\n".join(json.dumps(event) for event in parent_events), encoding="utf-8"
    )

    child_events = [
        {
            "type": "user",
            "isSidechain": True,
            "agentId": "child-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "sessionId": "parent-session",
            "version": "2.1.187",
            "message": {"role": "user", "content": "Explore code"},
        },
        {
            "type": "assistant",
            "isSidechain": True,
            "agentId": "child-1",
            "timestamp": "2026-01-01T00:00:01Z",
            "sessionId": "parent-session",
            "message": {
                "id": "child-message",
                "role": "assistant",
                "model": "claude-sonnet",
                "content": [{"type": "text", "text": "Child result"}],
                "usage": {
                    "input_tokens": 3,
                    "cache_read_input_tokens": 1,
                    "cache_creation_input_tokens": 2,
                    "output_tokens": 4,
                },
            },
        },
        {
            "type": "assistant",
            "isSidechain": True,
            "agentId": "child-1",
            "timestamp": "2026-01-01T00:00:02Z",
            "sessionId": "parent-session",
            "message": {
                "id": "child-agent-message",
                "role": "assistant",
                "model": "claude-sonnet",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-agent-2",
                        "name": "Agent",
                        "input": {"description": "Nested exploration"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "isSidechain": True,
            "agentId": "child-1",
            "timestamp": "2026-01-01T00:00:03Z",
            "sessionId": "parent-session",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-agent-2",
                        "content": "Nested agent launched",
                    }
                ],
            },
        },
    ]
    child_path = subagents_dir / "agent-child-1.jsonl"
    child_path.write_text(
        "\n".join(json.dumps(event) for event in child_events), encoding="utf-8"
    )
    child_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "agentType": "Explore",
                "description": "Explore code",
                "toolUseId": "tool-agent-1",
                "spawnDepth": 1,
            }
        ),
        encoding="utf-8",
    )
    grandchild_path = subagents_dir / "agent-grandchild-1.jsonl"
    grandchild_path.write_text(
        json.dumps(
            {
                "type": "assistant",
                "isSidechain": True,
                "agentId": "grandchild-1",
                "timestamp": "2026-01-01T00:00:03Z",
                "sessionId": "parent-session",
                "version": "2.1.187",
                "message": {
                    "id": "grandchild-message",
                    "role": "assistant",
                    "model": "claude-sonnet",
                    "content": [{"type": "text", "text": "Nested result"}],
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    grandchild_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "agentType": "Explore",
                "description": "Nested exploration",
                "toolUseId": "tool-agent-2",
                "spawnDepth": 2,
            }
        ),
        encoding="utf-8",
    )

    agent = ClaudeCode(logs_dir=logs_dir, model_name="claude-sonnet")
    parent = agent._convert_events_to_trajectory(session_dir)
    assert parent is not None

    agent._persist_subagent_trajectories(session_dir, parent)

    output_path = logs_dir / "subagents" / "agent-child-1.json"
    child = json.loads(output_path.read_text(encoding="utf-8"))
    assert child["trajectory_id"] == "agent-child-1"
    assert child["session_id"] == "parent-session"
    assert child["steps"][1]["extra"]["is_sidechain"] is True
    assert child["final_metrics"]["total_prompt_tokens"] == 6
    assert child["final_metrics"]["total_completion_tokens"] == 4
    assert "total_cost_usd" not in child["final_metrics"]

    ref = parent.steps[0].observation.results[0].subagent_trajectory_ref[0]
    assert ref.trajectory_path == "subagents/agent-child-1.json"
    assert ref.trajectory_id == "agent-child-1"
    assert ref.extra == {
        "agent_id": "child-1",
        "agent_type": "Explore",
        "description": "Explore code",
        "spawn_depth": 1,
    }
    nested_ref = child["steps"][2]["observation"]["results"][0][
        "subagent_trajectory_ref"
    ][0]
    assert nested_ref["trajectory_path"] == "subagents/agent-grandchild-1.json"
    assert nested_ref["extra"]["spawn_depth"] == 2
