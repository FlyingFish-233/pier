import json
from pathlib import Path

from pier.agents.installed.codex import Codex


PARENT_ID = "019fb2fb-2beb-7510-bbae-bcb7a98cde77"
CHILD_ID = "019fb2fb-66f9-7092-8072-6b79052972df"


def write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )


def session_meta(thread_id: str, **extra: object) -> dict:
    return {
        "timestamp": "2026-07-30T12:00:00Z",
        "type": "session_meta",
        "payload": {
            "id": thread_id,
            "cli_version": "0.145.0",
            "cwd": "/app",
            **extra,
        },
    }


def message(role: str, text: str) -> dict:
    return {
        "timestamp": "2026-07-30T12:00:01Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "output_text", "text": text}],
        },
    }


def test_persist_codex_subagent_trajectory_and_link(tmp_path: Path) -> None:
    logs_dir = tmp_path / "agent"
    sessions_dir = logs_dir / "sessions" / "2026" / "07" / "30"
    child_path = sessions_dir / "a-child.jsonl"
    parent_path = sessions_dir / "z-parent.jsonl"

    write_jsonl(
        child_path,
        [
            session_meta(
                CHILD_ID,
                session_id=PARENT_ID,
                parent_thread_id=PARENT_ID,
                thread_source="subagent",
                agent_nickname="Noether",
                source={
                    "subagent": {
                        "thread_spawn": {
                            "parent_thread_id": PARENT_ID,
                            "depth": 1,
                            "agent_nickname": "Noether",
                        }
                    }
                },
            ),
            message("user", "Inspect the repository"),
            message("assistant", "Inspection complete"),
        ],
    )
    write_jsonl(
        parent_path,
        [
            session_meta(PARENT_ID, thread_source="user"),
            message("user", "Delegate this task"),
            {
                "timestamp": "2026-07-30T12:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "spawn_agent",
                    "call_id": "spawn-call",
                    "arguments": json.dumps({"message": "Inspect the repository"}),
                },
            },
            {
                "timestamp": "2026-07-30T12:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "spawn-call",
                    "output": json.dumps({"agent_id": CHILD_ID, "nickname": "Noether"}),
                },
            },
        ],
    )

    agent = Codex(logs_dir=logs_dir, model_name="gpt-5.4-mini")
    trajectory = agent._convert_events_to_trajectory(logs_dir / "sessions")

    assert trajectory is not None
    assert trajectory.session_id == PARENT_ID
    assert agent._persist_subagent_trajectories(logs_dir / "sessions", trajectory) == 1

    output_path = logs_dir / "subagents" / f"agent-{CHILD_ID}.json"
    child_trajectory = json.loads(output_path.read_text(encoding="utf-8"))
    assert child_trajectory["session_id"] == CHILD_ID
    assert child_trajectory["trajectory_id"] == f"agent-{CHILD_ID}"

    refs = [
        ref
        for step in trajectory.steps
        if step.observation is not None
        for result in step.observation.results
        for ref in (result.subagent_trajectory_ref or [])
    ]
    assert len(refs) == 1
    assert refs[0].trajectory_id == f"agent-{CHILD_ID}"
    assert refs[0].trajectory_path == f"subagents/agent-{CHILD_ID}.json"
    assert refs[0].extra == {
        "agent_id": CHILD_ID,
        "nickname": "Noether",
        "parent_thread_id": PARENT_ID,
        "spawn_depth": 1,
    }
