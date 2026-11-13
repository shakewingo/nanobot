"""Dream cannot see the tail of a cumulative compaction checkpoint.

Each compaction asks the model to update the previous checkpoint
(`templates/agent/consolidator_archive.md`), and `MemoryArchiver.archive`
appends the whole result to `history.jsonl` (`agent/memory.py:955`).
`build_dream_prompt` then keeps only the first 1000 characters of every entry
(`agent/memory.py:521`), so facts past that point never reach Dream.

The model is scripted; everything from `archive()` to `build_dream_prompt()`
is real code. The control test feeds the same 40 facts as incremental
checkpoints (the pre-#5610 shape) and passes; the cumulative test fails.

Run:
    pytest tests/agent/test_dream_cumulative_truncation.py -q
"""

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import Consolidator, MemoryStore
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.utils.llm_runtime import LLMRuntime

SESSION_KEY = "telegram:demo"
FACTS_PER_COMPACTION = 10
COMPACTIONS = 4
FACTS = [
    f"- [durable] fact-{i:02d}: the user prefers configuration option {i} for subsystem {i}"
    for i in range(FACTS_PER_COMPACTION * COMPACTIONS)
]


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


@pytest.fixture
def provider():
    p = MagicMock(aclose=AsyncMock())
    p.chat_stream_with_retry = AsyncMock()
    p.generation = GenerationSettings(max_tokens=4096)
    return p


@pytest.fixture
def consolidator(store):
    return Consolidator(
        store=store,
        sessions=MagicMock(),
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
    )


async def _compact(consolidator, provider, checkpoint: str, previous: str | None) -> None:
    """Run one real archive call whose scripted model returns *checkpoint*."""
    provider.chat_stream_with_retry.return_value = LLMResponse(content=checkpoint)
    messages = [
        {"role": "user", "content": "conversation turn"},
        {"role": "assistant", "content": "ok"},
    ]
    await consolidator.archiver.archive(
        messages,
        runtime=LLMRuntime.capture(provider, "test-model", context_window_tokens=32_000),
        session_key=SESSION_KEY,
        history=[{"role": "system", "content": "system prompt"}, *messages],
        request_tools=[],
        previous_summary=previous,
    )


def _facts_seen_by_dream(store: MemoryStore) -> list[int]:
    prompt, _cursor = store.build_dream_prompt()
    return sorted({int(n) for n in re.findall(r"fact-(\d{2}):", prompt)})


async def test_incremental_checkpoints_reach_dream(store, provider, consolidator):
    for c in range(COMPACTIONS):
        new = FACTS[c * FACTS_PER_COMPACTION:(c + 1) * FACTS_PER_COMPACTION]
        await _compact(consolidator, provider, "\n".join(new), previous=None)

    assert _facts_seen_by_dream(store) == list(range(len(FACTS)))


async def test_cumulative_checkpoints_reach_dream(store, provider, consolidator):
    previous = None
    for c in range(COMPACTIONS):
        checkpoint = "\n".join(FACTS[:(c + 1) * FACTS_PER_COMPACTION])
        await _compact(consolidator, provider, checkpoint, previous)
        previous = checkpoint

    lengths = [len(e["content"]) for e in store.read_unprocessed_history(since_cursor=0)]
    seen = _facts_seen_by_dream(store)
    assert seen == list(range(len(FACTS))), (
        f"history entry lengths {lengths}; Dream sees only facts {seen}"
    )
