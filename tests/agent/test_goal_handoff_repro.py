"""#5256 on HEAD: an active goal turns one clarifying question into an unbounded loop.

Scenario from the issue: `/goal ...` records a goal, then the model needs an
answer from the user and replies with a question (no tool call). Since #6538
`GoalExecutionScope.continuation_prompt` (`goal/runtime.py:29`) re-prompts every
text response while the goal is active, and the runner loop has no cap for it
(`agent/runner.py:436-440`, `753-770`).

The model is scripted and keeps asking the same question. The only thing that
ends the run is this test's safety valve: at request SAFETY_VALVE the script
calls `update_goal(action="block")`. Without it the run would not terminate.

Run:
    python -m pytest tests/agent/test_goal_handoff_repro.py -q
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.goal.assembly import install_goal
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.session.goal_state import GOAL_STATE_KEY

MAX_ITERATIONS = 8
SAFETY_VALVE = 40
QUESTION = "Which day do you want to start the plan?"


def _provider(respond) -> MagicMock:
    provider = MagicMock(aclose=AsyncMock())
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.estimate_prompt_tokens.return_value = (100, "test")
    provider.chat_stream_with_retry = AsyncMock(side_effect=respond)
    return provider

# tmp_path is a built-in fixture in pytest
async def _run_one_message(tmp_path, provider, content: str):
    loop = AgentLoop(
        bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model",
        max_iterations=MAX_ITERATIONS, builtin_assembly=install_goal,
    )
    worker = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(InboundMessage(
            channel="cli", sender_id="user", chat_id="test", content=content,
        ))
        async with asyncio.timeout(10):
            while not loop._active_tasks.get("cli:test"):
                await asyncio.sleep(0)
            await asyncio.gather(*loop._active_tasks["cli:test"])
        session = await loop.sessions.state.get("cli:test")
        replies = sum(
            1 for m in session.messages
            if m.get("role") == "assistant" and m.get("content") == QUESTION
        )
        return session, replies
    finally:
        loop.stop()
        await asyncio.wait_for(worker, 5)


async def test_question_without_goal_ends_the_turn(tmp_path):
    provider = _provider(lambda **_: LLMResponse(content=QUESTION))

    _session, replies = await _run_one_message(tmp_path, provider, "plan my study schedule")

    assert provider.chat_stream_with_retry.await_count == 1
    assert replies == 1


async def test_question_with_active_goal_hands_back_to_user(tmp_path):
    calls = 0

    async def respond(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="create", name="create_goal",
                arguments={"objective": "Write a two-week Rust study plan to rust-plan.md."},
            )])
        if calls < SAFETY_VALVE:
            return LLMResponse(content=QUESTION)
        if calls == SAFETY_VALVE:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="valve", name="update_goal",
                arguments={"action": "block", "recap": "test safety valve"},
            )])
        return LLMResponse(content="Stopped.")

    session, replies = await _run_one_message(
        tmp_path, _provider(respond), "/goal plan a two-week Rust study schedule",
    )

    assert session.metadata[GOAL_STATE_KEY]["status"] == "blocked"
    assert calls <= 3, (
        f"max_iterations={MAX_ITERATIONS}; the goal run made {calls} requests and stored "
        f"{replies} copies of the same question before the test's safety valve stopped it"
    )
