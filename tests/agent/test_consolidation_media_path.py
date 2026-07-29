"""Uploaded media paths must survive session consolidation, not just LLM replay.

Regression test for the consolidation half of issue #5028. ``MemoryStore.
_format_messages`` (the summarizer input) used to read only ``content`` and skip
empty-content turns, so an uploaded file's absolute path — carried in the
``media`` field by 6 of 17 channels — was silently dropped at consolidation,
while ``Session.get_history`` preserved it on replay. Both renderers now share
``content_with_media_breadcrumbs``, so they agree about the path.

``_CHANNEL_SHAPES`` below is the executable form of the per-channel audit in the
issue write-up: each row is the persisted ``(content, media[])`` an uploaded
image produces for a channel, per a read of its inbound content-assembly code.
``path_in_content`` records whether the channel already inlines the ABSOLUTE
path into ``content`` (SAFE) or leaves it only in ``media[]`` (VULNERABLE — the
class this fix repairs).

Scope note: this file guards the shared *renderer* contract — that
``get_history`` (replay) and ``_format_messages`` (consolidation) agree about
``media[]`` for every audited shape. That a channel actually *emits* the shape
below is the responsibility of that channel's own tests under
``nanobot/channels/<channel>/tests/``; this file deliberately does not re-drive
channel runtimes.
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import Consolidator, MemoryStore, _missing_media_breadcrumbs
from nanobot.providers.base import GenerationSettings
from nanobot.session.manager import Session
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template

MEDIA_PATH = "/home/user/.nanobot/media/upload_photo.png"
MEDIA_NAME = "upload_photo.png"
TIMESTAMP = "2026-07-27T10:00"
MANIFEST_LINE = f"- [skip] Uploaded media available at [image: {MEDIA_PATH}]"


@dataclass(frozen=True)
class Shape:
    """A channel's persisted ``(content, media[])`` for an uploaded image."""

    channel: str
    content: str  # ``{P}`` = absolute path, ``{N}`` = basename
    path_in_content: bool  # True == SAFE (path already inlined); False == VULNERABLE
    source: str  # audited code reference
    has_media_field: bool = True  # a few channels persist no ``media[]`` at all


# msteams / mochat are intentionally absent: they wire no inbound media at all
# (text-only handler), so there is no uploaded path to preserve either way.
_CHANNEL_SHAPES = [
    # --- SAFE (9): absolute path inlined in content --------------------------
    Shape("telegram", "[image: {P}]", True, "telegram/runtime.py:1272,1521"),
    Shape("weixin", "[image]\n[Image: source: {P}]", True, "weixin/runtime.py:726"),
    Shape("feishu", "[image: {P}]", True, "feishu/runtime.py:1828"),
    Shape("signal", "[image: {P}]", True, "signal/runtime.py:923"),
    Shape("email", "[attachment: {N} — saved to {P}]", True, "email/runtime.py:534"),
    Shape("dingtalk", "[Image]\n\nReceived files:\n- {P}", True,
          "dingtalk/runtime.py:135-137", has_media_field=False),
    Shape("whatsapp", "[image: {P}]", True, "whatsapp/runtime.py:619,708"),
    Shape("matrix", "[attachment: {P}]", True, "matrix/runtime.py:1033"),
    Shape("qq", "[Image]\nReceived files:\n- {N}\n  saved: {P}", True, "qq/runtime.py:642,571"),
    # --- VULNERABLE (a): path only in media[]; content is bare user text -----
    Shape("websocket_image_only", "", False, "websocket/runtime.py:724 (image-only)"),
    Shape("websocket_caption", "please look", False, "websocket/runtime.py:724 (w/ caption)"),
    Shape("mattermost", "please look", False, "mattermost/runtime.py:265,279"),
    Shape("napcat", "please look", False, "napcat/runtime.py:271,283"),
    # --- VULNERABLE (b): basename/marker inlined, but not the path -----------
    Shape("slack", "please look\n[image: {N}]", False, "slack/runtime.py:434,467"),
    Shape("discord", "please look\n[attachment: {N}]", False, "discord/runtime.py:705"),
    Shape("wecom", "[image: {N}]", False, "wecom/runtime.py:275"),
]


def _message(shape: Shape) -> dict:
    content = shape.content.format(P=MEDIA_PATH, N=MEDIA_NAME)
    message = {"role": "user", "content": content, "timestamp": TIMESTAMP}
    if shape.has_media_field:
        message["media"] = [MEDIA_PATH]
    return message


def _replay_text(message: dict) -> str:
    history = Session(key="test-media-path", messages=[message]).get_history()
    return "\n".join(m.get("content", "") for m in history if isinstance(m.get("content"), str))


@pytest.mark.parametrize("shape", _CHANNEL_SHAPES, ids=lambda s: s.channel)
def test_media_path_survives_replay_and_consolidation(shape):
    """For every audited channel the absolute path reaches BOTH renderers, and
    the two never disagree about ``media[]`` — the #5028 invariant."""
    message = _message(shape)

    # Sanity: the constructed content matches the audited classification.
    assert (MEDIA_PATH in message["content"]) == shape.path_in_content

    in_replay = MEDIA_PATH in _replay_text(message)
    in_summary = MEDIA_PATH in MemoryStore._format_messages([message])

    assert in_replay, f"{shape.channel}: path lost on replay (get_history)"
    assert in_summary == in_replay, f"{shape.channel}: renderers disagree about media[]"


@pytest.mark.parametrize("path", [
    "/home/user/.nanobot/media/websocket/report.pdf",
    "/home/user/.nanobot/media/telegram/voice.ogg",
    "/home/user/.nanobot/media/websocket/clip.mp4",
], ids=["pdf", "audio", "video"])
def test_non_image_media_path_survives(path):
    """``media[]`` carries documents/audio/video too (not only images); their
    paths survive consolidation as well. The breadcrumb label is ``[image: …]``
    regardless of type — a pre-existing cosmetic trait of the shared helper —
    but the recoverable path is what matters here."""
    message = {"role": "user", "content": "", "media": [path], "timestamp": TIMESTAMP}
    assert path in MemoryStore._format_messages([message])


def test_format_messages_keeps_image_only_turn():
    """An image-only user turn (empty content, path only in ``media``) must not
    be skipped by ``_format_messages`` — its breadcrumb reaches the summarizer."""
    message = {"role": "user", "content": "", "media": [MEDIA_PATH], "timestamp": TIMESTAMP}
    assert f"[image: {MEDIA_PATH}]" in MemoryStore._format_messages([message])


# --- End-to-end: the path actually reaches the LLM summarizer via archive() ---


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    p.chat_with_retry.return_value = MagicMock(content="Summary.", finish_reason="stop")
    p.generation = GenerationSettings(max_tokens=100)
    return p


@pytest.fixture
def runtime(mock_provider):
    return LLMRuntime.capture(mock_provider, "test-model", context_window_tokens=1000)


@pytest.fixture
def consolidator(store):
    sessions = MagicMock()
    sessions.save = MagicMock()
    return Consolidator(
        store=store,
        sessions=sessions,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
    )


async def test_archive_preserves_media_path_with_caption(consolidator, mock_provider, runtime):
    """A user turn with a caption + ``media`` is summarized WITH the path: the
    summarizer input keeps both the caption and the file location."""
    await consolidator.archive(
        [{"role": "user", "content": "please look at this photo", "media": [MEDIA_PATH]}],
        runtime=runtime,
    )
    prompt = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
    assert "please look at this photo" in prompt
    assert MEDIA_PATH in prompt


async def test_archive_preserves_media_path_for_image_only_turn(
    consolidator, mock_provider, runtime
):
    """An image-only turn (empty content, path only in ``media``) still reaches
    the summarizer instead of vanishing from the input entirely."""
    await consolidator.archive(
        [
            {"role": "user", "content": "", "media": [MEDIA_PATH]},
            {"role": "assistant", "content": "nice photo!"},
        ],
        runtime=runtime,
    )
    prompt = mock_provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
    assert MEDIA_PATH in prompt


# --- Determinism: the path survives even when the summarizer omits it ---


def _entry(store) -> str:
    """The single history.jsonl entry written by an archive() call."""
    entries = store.read_unprocessed_history(since_cursor=0)
    assert len(entries) == 1
    return entries[0]["content"]


async def test_archive_restates_path_when_summary_omits_it(
    consolidator, mock_provider, store, runtime
):
    """The summarizer paraphrased the turn away; the manifest puts the path back."""
    mock_provider.chat_with_retry.return_value = MagicMock(
        content="User uploaded a photo and asked about it.",
        finish_reason="stop",
    )
    result = await consolidator.archive(
        [{"role": "user", "content": "", "media": [MEDIA_PATH]}],
        runtime=runtime,
    )
    assert _entry(store).startswith(MANIFEST_LINE)
    assert MEDIA_PATH in result


async def test_archive_does_not_duplicate_path_kept_by_summary(
    consolidator, mock_provider, store, runtime
):
    """A summarizer that already wrote the path gets no redundant manifest line."""
    mock_provider.chat_with_retry.return_value = MagicMock(
        content=f"User uploaded {MEDIA_PATH} and asked about it.",
        finish_reason="stop",
    )
    await consolidator.archive(
        [{"role": "user", "content": "", "media": [MEDIA_PATH]}],
        runtime=runtime,
    )
    assert _entry(store).count(MEDIA_PATH) == 1


async def test_archive_manifest_survives_an_oversized_summary(
    consolidator, mock_provider, store, runtime
):
    """The entry cap truncates from the end, so a leading manifest is untouchable."""
    mock_provider.chat_with_retry.return_value = MagicMock(
        content="x" * 20_000,
        finish_reason="stop",
    )
    await consolidator.archive(
        [{"role": "user", "content": "", "media": [MEDIA_PATH]}],
        runtime=runtime,
    )
    assert _entry(store).startswith(MANIFEST_LINE)


async def test_archive_manifest_covers_only_removed_messages(
    consolidator, mock_provider, store, runtime
):
    """Retained turns keep their own breadcrumb via get_history; only removed
    turns need restating."""
    retained_path = "/home/user/.nanobot/media/still_live.png"
    archived = {"role": "user", "content": "old", "media": [MEDIA_PATH]}
    retained = {"role": "user", "content": "new", "media": [retained_path]}
    await consolidator.archive(
        [archived],
        runtime=runtime,
        summary_messages=[archived, retained],
    )
    entry = _entry(store)
    assert MEDIA_PATH in entry
    assert retained_path not in entry


async def test_archive_scrubs_think_leak_without_losing_the_manifest(
    consolidator, mock_provider, store, runtime
):
    """A leading unclosed <think> must not reach history.jsonl, and stripping it
    must not take the manifest with it."""
    mock_provider.chat_with_retry.return_value = MagicMock(
        content="<think>internal reasoning about the user",
        finish_reason="stop",
    )
    result = await consolidator.archive(
        [{"role": "user", "content": "", "media": [MEDIA_PATH]}],
        runtime=runtime,
    )
    entry = _entry(store)
    assert entry == MANIFEST_LINE
    assert "internal reasoning" not in entry
    assert "internal reasoning" not in result


# --- Manifest helper: which archived media paths the summary failed to keep ---


def test_missing_breadcrumbs_reports_path_absent_from_summary():
    messages = [{"role": "user", "content": "", "media": [MEDIA_PATH]}]
    assert _missing_media_breadcrumbs(messages, "User sent a photo.") == MANIFEST_LINE


def test_missing_breadcrumbs_use_a_mark_the_consolidator_template_defines():
    """The manifest borrows the template's ``[skip]`` mark so Dream's per-mark
    rules apply to it; renaming the mark upstream must fail loudly here."""
    assert "[skip]" in render_template("agent/consolidator_archive.md", strip=True)


def test_missing_breadcrumbs_skips_path_already_in_summary():
    messages = [{"role": "user", "content": "", "media": [MEDIA_PATH]}]
    assert _missing_media_breadcrumbs(messages, f"Saved to {MEDIA_PATH} for later.") == ""


def test_missing_breadcrumbs_deduplicates_repeated_path():
    messages = [
        {"role": "user", "content": "one", "media": [MEDIA_PATH]},
        {"role": "user", "content": "two", "media": [MEDIA_PATH]},
    ]
    assert _missing_media_breadcrumbs(messages, "Summary.") == MANIFEST_LINE


def test_missing_breadcrumbs_ignores_non_user_roles():
    messages = [{"role": "assistant", "content": "here", "media": [MEDIA_PATH]}]
    assert _missing_media_breadcrumbs(messages, "Summary.") == ""


def test_missing_breadcrumbs_tolerates_malformed_media():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b", "media": None},
        {"role": "user", "content": "c", "media": ["", None, 7]},
    ]
    assert _missing_media_breadcrumbs(messages, "Summary.") == ""
