"""Share one ``AcpRuntime`` process across several top-level chat sessions.

Subagents already multiplex sessions onto their parent's process. This module
extends the same demux to top-level dashboard chat slots, which otherwise spawn
one ``kiro-cli`` process each.

Three rules govern it, and each exists because breaking it is silently wrong
rather than loudly wrong:

``eligible_for_chat_sharing``
    Only a dashboard chat slot running in persistent memory mode may share. A
    cron, hook, task-runner or crew-member session keeps its own process, and so
    does an incognito or temporary session -- see the function's own reasoning.

:class:`ChatRuntimeKey`
    Two sessions may land on one process only when every PROCESS-LEVEL spawn
    input matches. A per-session input (the ACP ``cwd``, the session key, the
    crew identity) is deliberately absent from the key, because ``create_session``
    carries it per session. An input whose scope is ambiguous is IN the key: a
    key that is too narrow costs memory, while one that is too wide runs a
    session under another session's process configuration.

:class:`ChatRuntimePool`
    A runtime lives while at least one session holds it. The last release kills
    it; every earlier release only drops a reference. The cap bounds how many
    sessions one process serves, which is the blast radius of that process
    dying.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from kiro_crew.messaging.link import telemetry_channel_of

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kiro_crew.acp.runtime import AcpRuntime
    from kiro_crew.agent_sdk.tool_search import ToolSearchSettings

logger = logging.getLogger(__name__)

#: Memory modes that may share a process. A shared process OUTLIVES any single
#: session on it, because the other sessions hold the reference -- so an
#: incognito or temporary session's teardown cannot take the process (and the
#: scratch files it wrote) with it, which is exactly what that mode promises.
#: ``create_session`` also latches ``recording_allowed`` off for the WHOLE
#: runtime when it starts a non-persistent session, so one such session joining
#: would stop recording for every other session on the process.
_SHAREABLE_MEMORY_MODES = frozenset({"persistent"})


def eligible_for_chat_sharing(
    *,
    session_key: str | None,
    memory_mode: str,
    member_context: bool,
    sharing_enabled: bool,
) -> bool:
    """Whether this session may join (or found) a shared chat runtime.

    Scoped to dashboard chat slots. ``telemetry_channel_of`` is the repository's
    single classification of a session key's origin, and it answers
    ``"dashboard"`` only for a chat slot -- a cron, hook, task-runner, subagent
    or channel-bound key classifies as something else, so each keeps its own
    process without this function naming any of them.

    ``member_context`` is refused because a crew member's process captures that
    member's native launch documents at spawn, so its process is already
    member-specific.
    """
    if not sharing_enabled:
        return False
    if member_context:
        return False
    if memory_mode not in _SHAREABLE_MEMORY_MODES:
        return False
    return telemetry_channel_of(session_key) == "dashboard"


def _freeze_env(extra_env: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    """A hashable, order-independent form of the child's extra environment."""
    if not extra_env:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in extra_env.items()))


def _freeze_path(value: str | Path | None) -> str:
    """A comparable spelling of a path-shaped spawn input (``""`` when unset).

    Both spellings of one directory have to land on the same key, and a bare
    ``str()`` does not deliver that: on Windows a ``Path`` stringifies with
    backslashes while a caller's plain string keeps the separators it was
    written with, so the same work directory would key two processes and the
    pool would spawn instead of sharing. Coercing through ``Path`` first gives
    one spelling per platform, and it also folds a trailing separator, which is
    the same directory by every other measure.

    Deliberately NOT ``resolve()``: this runs on the placement path for every
    session start, and resolving touches the filesystem and can raise. The key
    compares what the caller asked to spawn with, which is what the pre-spawn
    gates judge too.
    """
    if not value:
        return ""
    return str(Path(value))


def _freeze_tool_search(settings: "ToolSearchSettings | None") -> tuple[object, ...]:
    """The Tool Search choice as sent at ``initialize``, or ``()`` when unset.

    Sent once per process in ``clientCapabilities``, so a session cannot carry
    its own -- a session joining a process configured differently would run with
    a setting it did not ask for.
    """
    if settings is None:
        return ()
    return (
        bool(settings.enabled),
        int(settings.min_pct),
        int(settings.min_tokens),
    )


@dataclass(frozen=True)
class ChatRuntimeKey:
    """Every process-level spawn input of one ``AcpRuntime``.

    One line per field on why it cannot vary between two sessions on a process:

    ``work_dir``
        The child's own cwd, and the directory four pre-spawn gates judge:
        derived-spec freshness, fork governance, the sealed-target refusal and
        the voice-workspace assertion. ``kiro-cli`` also resolves ``--agent``
        against ``<work_dir>/.kiro/agents`` before the global directory, so the
        spec a session runs follows the process, not the session.
    ``agent``
        Carried on argv as ``--agent``, and rewritten in place by the native
        skill projection.
    ``model``
        Carried on argv as ``--model`` when pinned at spawn. A later per-session
        model selection is a session request and stays out of this key.
    ``sandbox_mode``
        Chooses the wrap, the credential mask and whether the host's internal
        sandbox is delegated to.
    ``extra_env``
        The child's environment, which one process has exactly one of.
    ``acp_backend``
        Selects the harness, and through it the binary and the handshake.
    ``tool_search``
        Sent once in the ``initialize`` handshake.
    ``member_context``
        Decides whether native launch documents are captured at spawn.
    ``memory_mode``
        Latches the runtime's recording permission at ``create_session``.
    ``shared_scratch``
        The session tree's work directory, mounted into the process.
    ``mcp_gateway_overlay`` / ``mcp_gateway_socket``
        Held on the runtime and read when composing each session's MCP array.
        Scope is ambiguous, so both are in the key.
    ``expect_mcp_reports``
        Whether sessions on this process wait for the MCP readiness ceiling.
    ``max_age_secs`` / ``max_rss_mb``
        The recycle thresholds the process is governed by.
    """

    work_dir: str
    agent: str
    model: str
    sandbox_mode: str
    extra_env: tuple[tuple[str, str], ...]
    acp_backend: str
    tool_search: tuple[object, ...]
    member_context: bool
    memory_mode: str
    shared_scratch: str
    mcp_gateway_overlay: str
    mcp_gateway_socket: str
    expect_mcp_reports: bool
    max_age_secs: float
    max_rss_mb: float

    @classmethod
    def build(
        cls,
        *,
        work_dir: str | Path | None,
        agent: str,
        sandbox_mode: str,
        extra_env: dict[str, str] | None,
        acp_backend: str,
        tool_search: "ToolSearchSettings | None",
        member_context: bool,
        memory_mode: str,
        shared_scratch: Path | None,
        mcp_gateway_overlay: str | Path | None,
        mcp_gateway_socket: str | Path | None,
        model: str | None = None,
        expect_mcp_reports: bool = True,
        max_age_secs: float = 0.0,
        max_rss_mb: float = 0.0,
    ) -> "ChatRuntimeKey":
        """Build the key from the values a caller is about to spawn with.

        Keyword-only and exhaustive on purpose: a field added to the runtime's
        constructor and forgotten here would widen the key silently, which is
        the failure this shape makes impossible to do by accident.
        """
        return cls(
            work_dir=_freeze_path(work_dir),
            agent=agent or "",
            model=model or "",
            sandbox_mode=sandbox_mode or "",
            extra_env=_freeze_env(extra_env),
            acp_backend=acp_backend or "",
            tool_search=_freeze_tool_search(tool_search),
            member_context=bool(member_context),
            memory_mode=memory_mode or "",
            shared_scratch=_freeze_path(shared_scratch),
            mcp_gateway_overlay=_freeze_path(mcp_gateway_overlay),
            mcp_gateway_socket=_freeze_path(mcp_gateway_socket),
            expect_mcp_reports=bool(expect_mcp_reports),
            max_age_secs=float(max_age_secs),
            max_rss_mb=float(max_rss_mb),
        )


@dataclass
class _Entry:
    """One live runtime and the session keys holding it."""

    runtime: "AcpRuntime"
    key: ChatRuntimeKey
    refs: set[str] = field(default_factory=set)

    def has_room(self, cap: int) -> bool:
        return len(self.refs) < cap


@dataclass
class Acquisition:
    """What :meth:`ChatRuntimePool.acquire` handed back.

    ``joined`` is False for the session that FOUNDED the runtime and True for
    every session that landed on an existing one. Callers use it for the two
    decisions that differ between the two cases: whether the provider owns the
    process, and whether per-session start work that rewrites process-level
    state may run.
    """

    runtime: "AcpRuntime"
    joined: bool
    sessions_on_runtime: int


class ChatRuntimePool:
    """Registry of shared chat runtimes, keyed by process-level compatibility.

    One lock serializes the whole registry rather than one lock per key. The
    critical section is a dict lookup plus, for a miss, one spawn -- and a spawn
    already serializes behind the gateway-wide cold-start admission gate, so a
    finer lock would buy concurrency the spawn path does not have.
    """

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._by_session: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    # ── reads ──

    def runtime_for(self, session_key: str) -> "AcpRuntime | None":
        """The runtime *session_key* holds, or None when it holds none."""
        entry = self._by_session.get(session_key)
        return entry.runtime if entry is not None else None

    def sessions_on(self, runtime: "AcpRuntime") -> list[str]:
        """Every session key holding *runtime*.

        The answer any PID-to-session consumer needs once one process serves
        several top-level sessions: a process identifies a GROUP of sessions
        rather than one, so the mapping has to be read rather than inferred.
        """
        for entry in self._entries:
            if entry.runtime is runtime:
                return sorted(entry.refs)
        return []

    def shared_pids(self) -> dict[int, list[str]]:
        """Live shared process ids mapped to the session keys they serve."""
        rows: dict[int, list[str]] = {}
        for entry in self._entries:
            pid = getattr(entry.runtime, "pid", None)
            if isinstance(pid, int) and pid > 0:
                rows[pid] = sorted(entry.refs)
        return rows

    # ── writes ──

    async def acquire(
        self,
        key: ChatRuntimeKey,
        session_key: str,
        spawn: Callable[[], Awaitable["AcpRuntime"]],
        *,
        cap: int,
        prefer_session_key: str | None = None,
    ) -> Acquisition:
        """Hold a runtime for *session_key*, spawning one only when needed.

        Placement order, which is the placement decision in full:

        1. ``prefer_session_key``'s runtime -- the root of this session's tree --
           when its key matches and it has room. A session started from a root
           shares that root's process so the tree stays on one process where it
           can.
        2. Otherwise the first compatible runtime with room.
        3. Otherwise ``spawn()``.

        A dead runtime is never handed out and never counted: it is dropped on
        the way past, so the caller that finds none alive spawns, and the
        sessions that were on it rejoin through this same path on their own next
        turn.
        """
        async with self._lock:
            self._drop_dead_locked()

            existing = self._by_session.get(session_key)
            if existing is not None:
                return Acquisition(
                    runtime=existing.runtime,
                    joined=True,
                    sessions_on_runtime=len(existing.refs),
                )

            entry = self._pick_locked(key, cap=cap, prefer_session_key=prefer_session_key)
            if entry is not None:
                entry.refs.add(session_key)
                self._by_session[session_key] = entry
                logger.info(
                    "chat_runtime_sharing outcome=joined pid=%s sessions=%d cap=%d",
                    getattr(entry.runtime, "pid", None),
                    len(entry.refs),
                    cap,
                )
                return Acquisition(
                    runtime=entry.runtime,
                    joined=True,
                    sessions_on_runtime=len(entry.refs),
                )

            runtime = await spawn()
            entry = _Entry(runtime=runtime, key=key, refs={session_key})
            self._entries.append(entry)
            self._by_session[session_key] = entry
            logger.info(
                "chat_runtime_sharing outcome=spawned pid=%s runtimes=%d cap=%d",
                getattr(runtime, "pid", None),
                len(self._entries),
                cap,
            )
            return Acquisition(runtime=runtime, joined=False, sessions_on_runtime=1)

    async def release(self, session_key: str) -> "AcpRuntime | None":
        """Drop *session_key*'s reference; return the runtime to kill, if any.

        The runtime is returned rather than killed here so the caller performs
        the teardown it already performs, with its own logging and error
        handling. Returns None while any other session still holds it, which is
        what stops one session's close, reset or model switch from killing a
        process other sessions are mid-turn on.
        """
        async with self._lock:
            entry = self._by_session.pop(session_key, None)
            if entry is None:
                return None
            entry.refs.discard(session_key)
            if entry.refs:
                logger.info(
                    "chat_runtime_sharing outcome=released pid=%s remaining=%d",
                    getattr(entry.runtime, "pid", None),
                    len(entry.refs),
                )
                return None
            self._forget_entry_locked(entry)
            logger.info(
                "chat_runtime_sharing outcome=last_release pid=%s",
                getattr(entry.runtime, "pid", None),
            )
            return entry.runtime

    async def forget(self, runtime: "AcpRuntime") -> list[str]:
        """Unbind every session from *runtime* and return their keys.

        Called when a runtime is known dead. Each returned session recovers on
        its own next turn through the ordinary start path, which finds no
        runtime bound and acquires one -- the first to arrive spawns, the rest
        join it.
        """
        async with self._lock:
            for entry in list(self._entries):
                if entry.runtime is runtime:
                    orphaned = sorted(entry.refs)
                    self._forget_entry_locked(entry)
                    if orphaned:
                        logger.warning(
                            "chat_runtime_sharing outcome=runtime_lost pid=%s sessions=%d: "
                            "each recovers on its next turn",
                            getattr(runtime, "pid", None),
                            len(orphaned),
                        )
                    return orphaned
            return []

    # ── internals ──

    def _pick_locked(
        self,
        key: ChatRuntimeKey,
        *,
        cap: int,
        prefer_session_key: str | None,
    ) -> _Entry | None:
        if prefer_session_key:
            preferred = self._by_session.get(prefer_session_key)
            if (
                preferred is not None
                and preferred.key == key
                and preferred.has_room(cap)
                and self._alive(preferred)
            ):
                return preferred
        for entry in self._entries:
            if entry.key == key and entry.has_room(cap) and self._alive(entry):
                return entry
        return None

    @staticmethod
    def _alive(entry: _Entry) -> bool:
        probe = getattr(entry.runtime, "is_alive", None)
        if probe is None:
            return True
        try:
            return bool(probe())
        except Exception:
            return False

    def _drop_dead_locked(self) -> None:
        for entry in list(self._entries):
            if not self._alive(entry):
                orphaned = sorted(entry.refs)
                self._forget_entry_locked(entry)
                if orphaned:
                    logger.warning(
                        "chat_runtime_sharing outcome=dead_runtime_dropped pid=%s sessions=%d",
                        getattr(entry.runtime, "pid", None),
                        len(orphaned),
                    )

    def _forget_entry_locked(self, entry: _Entry) -> None:
        for held in list(entry.refs):
            if self._by_session.get(held) is entry:
                del self._by_session[held]
        entry.refs.clear()
        self._entries = [e for e in self._entries if e is not entry]


#: The gateway's one chat runtime pool. A module-level singleton because the
#: sharing decision has to be global: two chat sessions can only land on one
#: process if both consult the same registry.
CHAT_RUNTIME_POOL = ChatRuntimePool()
