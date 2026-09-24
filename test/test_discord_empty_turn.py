"""A Discord turn that ends with no assistant text must never read as a reply.

The recorded incident shape: the backend closes the turn with a terminal
``end_turn`` and streams no text chunk at all. On that shape the pipeline used
to (1) edit the live ``…`` placeholder into ``…`` plus the ``Finished in …``
footer, which is indistinguishable from a finished answer, and (2) persist the
user's row alone, so the transcript held no trace that the model returned
nothing and the dashboard offered an "interrupted turn" recovery for a turn
that had in fact completed.

The sibling shape, seen the same day: the provider raised before any text,
after the user had steered mid-turn. The steer chip (a ``> quoted`` line) made
the body non-empty, so the error placeholder was skipped and the bubble closed
on the chip plus the footer -- and, because the exception escaped ahead of the
persist step, the transcript recorded neither the message nor the error.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_discord import (
    FakeClient,
    FakeCtx,
    FakeProvider,
    FakeSessions,
    _cfg,
    _Ev,
    _inbound,
    _prime_live,
)

from kiro_crew.acp.types import EVENT_COMPLETE, EVENT_TEXT_CHUNK, EVENT_TOOL_CALL
from kiro_crew.discord.renderer import DiscordRenderer
from kiro_crew.discord.transport import DISCORD_CAPABILITIES
from kiro_crew.discord.transport_dispatch import DiscordDispatcher
from kiro_crew.history import ConversationLog

FOOTER_MARK = "\n\n-# "


class _EmptyCompletionProvider(FakeProvider):
    """The incident's frame sequence: a terminal completion and nothing else."""

    async def stream(self, message: str) -> Any:
        yield _Ev(EVENT_COMPLETE, stop_reason="end_turn")


class _ToolOnlyProvider(FakeProvider):
    """A turn that ran a tool and then closed without a closing reply."""

    async def stream(self, message: str) -> Any:
        yield _Ev(EVENT_TOOL_CALL, title="Read the config")
        yield _Ev(EVENT_COMPLETE, stop_reason="end_turn")


class _DyingProvider(FakeProvider):
    """The provider fails before any text lands (the backend-error shape)."""

    async def stream(self, message: str) -> Any:
        raise RuntimeError("The model failed to generate a response (transient error)")
        yield  # pragma: no cover -- makes this an async generator


class _WhitespaceReplyProvider(FakeProvider):
    """A turn whose only "text" is the separator a steer boundary emits."""

    async def stream(self, message: str) -> Any:
        yield _Ev(EVENT_TEXT_CHUNK, text="\n")
        yield _Ev(EVENT_COMPLETE, stop_reason="end_turn")


class _Sessions(FakeSessions):
    """``FakeSessions`` that hands the turn a chosen provider."""

    def __init__(self, provider: FakeProvider) -> None:
        super().__init__()
        self._provider = provider

    async def get_or_create(self, key: str, **kw: Any) -> Any:
        self.last_provider = self._provider
        return self._provider, True, False


def _dispatcher(
    provider: FakeProvider, log_dir: Path
) -> tuple[DiscordDispatcher, FakeClient, _Sessions, ConversationLog]:
    sess = _Sessions(provider)
    cfg = _cfg()
    _prime_live(cfg)
    conv_log = ConversationLog(base_dir=log_dir)
    d = DiscordDispatcher(
        sessions=sess,  # type: ignore[arg-type]
        ctx_builder=FakeCtx(),  # type: ignore[arg-type]
        cfg=cfg,
        allowed_user_ids={"u1"},
        allowed_thread_ids=None,
        agent=None,
        conv_log=conv_log,
    )
    cli = FakeClient()
    d.client = cli  # type: ignore[assignment]
    return d, cli, sess, conv_log


def _body(final_text: str) -> str:
    """The bubble's body without the ``-# Finished in …`` subtext footer."""
    return final_text.split(FOOTER_MARK, 1)[0].strip()


def _rows(conv_log: ConversationLog, d: DiscordDispatcher) -> list[dict]:
    return conv_log.read_messages(d._session_key("u1", ""))


class TestEmptyTurnIsNeverAFinishedReply:
    @pytest.mark.asyncio
    async def test_a_completed_turn_with_no_text_posts_a_notice_not_the_live_placeholder(
        self, tmp_path: Path
    ) -> None:
        d, cli, _sess, _log = _dispatcher(_EmptyCompletionProvider(), tmp_path)

        await d.handle_message(_inbound("how do I run ten coders at once?"))

        final = cli.final_text()
        assert final is not None
        body = _body(final)
        # The live placeholder is "…"; a turn that CLOSED with nothing must not
        # hand the user that same glyph under a "Finished in" footer.
        assert body != "…", f"the finished bubble is the live placeholder: {final!r}"
        assert body, f"the finished bubble carries only the footer: {final!r}"
        assert "returned nothing" in body

    @pytest.mark.asyncio
    async def test_the_empty_reply_is_recorded_in_the_transcript(self, tmp_path: Path) -> None:
        d, cli, sess, log = _dispatcher(_EmptyCompletionProvider(), tmp_path)

        await d.handle_message(_inbound("how do I run ten coders at once?"))

        rows = _rows(log, d)
        roles = [r["role"] for r in rows]
        # The user's row must not be the last word: a reader (or the dashboard's
        # recovery) needs the record that the turn completed with no reply.
        assert roles[:1] == ["user"]
        assert len(rows) >= 2, f"the transcript ends on the user's row: {roles}"
        assert rows[-1]["role"] == "notice"
        assert rows[-1]["content"] == _body(cli.final_text() or "")
        # The prompt reached the model and the turn closed, so the session's
        # health counter is untouched -- the notice is the outcome, not a fault.
        assert sess.successes and not sess.failures

    @pytest.mark.asyncio
    async def test_a_tool_only_turn_says_so_instead_of_claiming_nothing_happened(
        self, tmp_path: Path
    ) -> None:
        d, cli, _sess, log = _dispatcher(_ToolOnlyProvider(), tmp_path)

        await d.handle_message(_inbound("check the config"))

        body = _body(cli.final_text() or "")
        assert "without a closing reply" in body
        assert "returned nothing" not in body
        assert _rows(log, d)[-1]["content"] == body

    @pytest.mark.asyncio
    async def test_a_steer_chip_does_not_mask_a_turn_that_died_without_text(self) -> None:
        cli = FakeClient()
        r = DiscordRenderer(cli, "chan1", DISCORD_CAPABILITIES, session_key="sk")  # type: ignore[arg-type]
        await r.on_turn_start()
        r.note_steer("steer-me-42")
        # The turn never reached on_done: the dispatcher's finally closes it.
        await r.close()

        final = cli.final_text()
        assert final is not None
        body = _body(final)
        assert "steer-me-42" in body  # the user's own steer is still shown
        assert (
            body.replace("> ↪️ steer-me-42", "").replace("> steer-me-42", "").strip()
        ), f"the bubble is the steer chip alone under a finished footer: {final!r}"
        assert "⚠️" in body

    @pytest.mark.asyncio
    async def test_a_steer_burst_never_pushes_the_placeholder_past_the_platform_cut(self) -> None:
        """The chip rides on the placeholder, not through the length rotation, and
        the client cuts one payload at ``DISCORD_MAX_TEXT``. A burst of steers
        (each already capped by ``_neutralize_md``) must be bounded so the notice
        and the footer -- the sentence this path exists to deliver -- survive."""
        from kiro_crew.discord.client import DISCORD_MAX_TEXT

        cli = FakeClient()
        r = DiscordRenderer(cli, "chan1", DISCORD_CAPABILITIES, session_key="sk")  # type: ignore[arg-type]
        await r.on_turn_start()
        for i in range(50):
            r.note_steer(f"steer number {i:02d} " + "x" * 100)
        await r.close()

        final = cli.final_text()
        assert final is not None
        assert len(final) <= DISCORD_MAX_TEXT, len(final)
        assert "⚠️ Error" in final
        assert "-# Finished in" in final
        assert final.startswith("> steer number 00")  # the chip is cut, not dropped

    @pytest.mark.asyncio
    async def test_a_turn_that_died_before_any_text_is_recorded_with_its_error(
        self, tmp_path: Path
    ) -> None:
        d, cli, sess, log = _dispatcher(_DyingProvider(), tmp_path)

        await d.handle_message(_inbound("hello?"))

        assert sess.failures
        body = _body(cli.final_text() or "")
        assert "⚠️" in body
        rows = _rows(log, d)
        roles = [r["role"] for r in rows]
        assert roles == ["user", "error"], f"the failed turn left no record: {roles}"
        assert "failed to generate" in rows[-1]["content"]

    @pytest.mark.asyncio
    async def test_an_undelivered_notice_is_recorded_as_a_failure(self, tmp_path: Path) -> None:
        """The notice is the turn's ENTIRE delivery; if Discord never took it, the
        user heard nothing, and that is the undelivered turn `record_failure` is
        for -- not a success with an empty body."""
        d, _cli, sess, log = _dispatcher(_EmptyCompletionProvider(), tmp_path)
        _cli.edit_ok = False
        _cli.fail_sends = True

        await d.handle_message(_inbound("anyone there?"))

        assert sess.failures and not sess.successes
        # The record of what happened is still written: the transcript is the
        # one place a reader can learn the turn closed empty.
        assert [r["role"] for r in _rows(log, d)] == ["user", "notice"]

    @pytest.mark.asyncio
    async def test_a_whitespace_only_reply_files_no_assistant_row(self, tmp_path: Path) -> None:
        """The steer-boundary separator alone (``"\\n"``) is not a reply: the
        durable write files no assistant row and records the notice instead."""
        d, cli, _sess, log = _dispatcher(_WhitespaceReplyProvider(), tmp_path)

        await d.handle_message(_inbound("hm"))

        assert [r["role"] for r in _rows(log, d)] == ["user", "notice"]
        assert "returned nothing" in _body(cli.final_text() or "")


class TestLiveWindowAndDiskAgree:
    """A resumed dashboard session mirrors the turn into its open window first and
    then persists under the SAME row ids; the two writers must file the same rows
    for a textless turn, or the slot's own save lands a row the disk never got."""

    @staticmethod
    def _state_and_slot(tmp_path: Path) -> tuple[Any, Any]:
        from chat_test_helpers import _make_state

        state = _make_state(tmp_path)
        state.sessions.channel_key_for_stem = lambda stem: ""
        slot = state.get_or_create_slot(name="chat-1", linked_session_key="dashboard:chat-1")
        return state, slot

    @pytest.mark.parametrize("save_first", [True, False], ids=["save-first", "write-first"])
    def test_the_notice_lands_exactly_once_and_no_assistant_row_appears(
        self, tmp_path: Path, save_first: bool
    ) -> None:
        from kiro_crew.dashboard.channel_slots import (
            project_channel_row_live,
            project_channel_turn_live,
        )
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history
        from kiro_crew.dashboard.state import row_mid
        from kiro_crew.messaging.driver import EMPTY_TURN_NOTICE

        state, slot = self._state_and_slot(tmp_path)
        key = "dashboard:chat-1"
        # What the dispatcher hands both writers: the reply normalized ONCE to
        # "" (the turn streamed only a steer-boundary "\n"), and the verdict.
        mids = project_channel_turn_live(state, key, "hello", "")
        assert mids is not None
        notice_mid = project_channel_row_live(
            state, key, "notice", EMPTY_TURN_NOTICE, "msg msg-info"
        )
        assert notice_mid

        window = [(m["role"], m["content"]) for m in slot.messages]
        assert window == [("user", "hello"), ("notice", EMPTY_TURN_NOTICE)]

        if save_first:
            assert _save_slot_to_history(state, slot, force=True)
        DiscordDispatcher._persist_turn(
            SimpleNamespace(conv_log=state.conversation_log),  # type: ignore[arg-type]
            key,
            "hello",
            "",
            False,
            agent="kirocrew",
            mirror_mids=mids,
            extra_row=("notice", EMPTY_TURN_NOTICE, "msg msg-info", notice_mid),
        )
        if not save_first:
            assert _save_slot_to_history(state, slot, force=True)

        rows = state.conversation_log.read_messages(key)
        assert [(r["role"], r["content"]) for r in rows] == window
        assert [row_mid(r) for r in rows] == [mids[0], notice_mid]


class TestEmptyTurnVerdict:
    """The driver's verdict is the one source the bubble and the transcript share."""

    def _run(self, events: list[Any]) -> tuple[Any, Any, str]:
        import asyncio

        from test_messaging_driver import _RecordingRenderer, _ScriptedProvider

        from kiro_crew.messaging.driver import TurnDriver

        renderer = _RecordingRenderer()
        driver = TurnDriver(_ScriptedProvider(events), renderer)
        accumulated = asyncio.run(driver.run("hello"))
        return driver, renderer, accumulated

    def test_a_textless_end_turn_is_an_empty_reply(self) -> None:
        from kiro_crew.acp.types import AcpEvent
        from kiro_crew.messaging.driver import EMPTY_TURN_NOTICE

        driver, renderer, accumulated = self._run(
            [AcpEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")]
        )
        assert accumulated == ""
        assert driver.empty_turn_notice == EMPTY_TURN_NOTICE
        # The renderer learned the same verdict from the DONE event it was handed.
        assert renderer.empty_turn_notice == EMPTY_TURN_NOTICE

    def test_a_turn_that_produced_text_has_no_notice(self) -> None:
        from kiro_crew.acp.types import AcpEvent

        driver, renderer, _ = self._run(
            [
                AcpEvent(kind=EVENT_TEXT_CHUNK, text="Here you go."),
                AcpEvent(kind=EVENT_COMPLETE, stop_reason="end_turn"),
            ]
        )
        assert driver.empty_turn_notice == ""
        assert renderer.empty_turn_notice == ""

    def test_a_cancelled_turn_is_not_an_empty_reply(self) -> None:
        from kiro_crew.acp.types import AcpEvent

        driver, _renderer, _ = self._run([AcpEvent(kind=EVENT_COMPLETE, stop_reason="cancelled")])
        assert driver.empty_turn_notice == ""

    def test_a_tool_only_turn_takes_the_after_work_wording(self) -> None:
        from kiro_crew.acp.types import AcpEvent
        from kiro_crew.messaging.driver import EMPTY_TURN_NOTICE_AFTER_WORK

        driver, _renderer, _ = self._run(
            [
                AcpEvent(kind=EVENT_TOOL_CALL, tool_call_id="t1", title="Read the config"),
                AcpEvent(kind=EVENT_COMPLETE, stop_reason="end_turn"),
            ]
        )
        assert driver.empty_turn_notice == EMPTY_TURN_NOTICE_AFTER_WORK

    def test_an_error_family_terminal_names_its_reason(self) -> None:
        from kiro_crew.acp.types import STOP_REASON_TOOL_STALL, AcpEvent

        driver, _renderer, _ = self._run(
            [AcpEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_TOOL_STALL)]
        )
        assert driver.empty_turn_notice.startswith("⚠️")
        assert "tool stall" in driver.empty_turn_notice

    def test_a_backend_authored_error_reason_never_reaches_the_copy(self) -> None:
        """The ``error:`` family is open on the wire: a backend can put anything
        after the prefix. Only the closed protocol values are named; the rest
        take fixed generic copy, so no backend prose (a leaked token, a path)
        rides an unredacted notice into the channel and the transcript."""
        from kiro_crew.acp.types import AcpEvent

        wire = "error: AKIAZZZZEXAMPLEKEY000 at /srv/secret/path"
        driver, _renderer, _ = self._run([AcpEvent(kind=EVENT_COMPLETE, stop_reason=wire)])
        assert driver.empty_turn_notice.startswith("⚠️")
        assert "backend error" in driver.empty_turn_notice
        assert "AKIAZZZZ" not in driver.empty_turn_notice
        assert "/srv/" not in driver.empty_turn_notice

    def test_a_stream_that_never_closed_the_turn_is_an_empty_reply(self) -> None:
        from kiro_crew.acp.types import AcpEvent
        from kiro_crew.messaging.driver import EMPTY_TURN_NOTICE_UNCLOSED

        driver, renderer, _ = self._run([AcpEvent(kind=EVENT_TEXT_CHUNK, text="")])
        assert driver.completion_observed is False
        assert driver.empty_turn_notice == EMPTY_TURN_NOTICE_UNCLOSED
        # No DONE was dispatched, so the renderer learned nothing; its close()
        # path posts its own error placeholder and the dispatcher records this.
        assert renderer.empty_turn_notice == ""
