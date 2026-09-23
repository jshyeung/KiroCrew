"""Channel-neutral ``!temporary`` / ``!incognito`` session privacy modifiers.

Two modes, and what each one actually forbids:

Temporary (blank-slate)
    No memory reads, no memory writes, no persistence. The session starts with
    zero context and discards everything on close.
Incognito
    Memory reads are allowed but writes are blocked; the ephemeral conversation
    log is discarded on close.

Both are tracked in bounded LRUs keyed by **session key**, never by a platform
thread id. That is what lets one copy of the machinery serve a Slack thread ts,
a Telegram DM route and a Telegram forum Topic without any of them colliding —
and it is why :func:`is_restricted` answers correctly for a
``telegram:{agent}:direct:{user}`` key, which a ``startswith("slack:")`` test
never could. The LRUs are process-local; :func:`hydrate` rebuilds them from the
durable ``SessionMap`` flag so a mode survives a gateway restart.

What a second channel must supply
---------------------------------
Everything platform-shaped is a parameter, because this module may not import
``kiro_crew.slack`` or ``kiro_crew.dashboard`` — not even function-locally (the
one-way dependency invariant in ``docs/system-specs/modules/messaging.md``):

* ``source`` — the channel name, e.g. ``"slack"``. It is the audit label and
  nothing else: the SEL operation is ``f"{source}.{mode}_mode"`` and the event's
  ``source`` field is ``source`` verbatim.
* ``sessions`` — the ``SessionManager``. Supplying it makes the mode DURABLE, in
  two records for two readers: the per-conversation flag in the one
  :class:`~kiro_crew.session_map.SessionMap` instance it owns (so the durable
  flag cannot be clobbered by a second instance's save), which is what
  :func:`hydrate` restores the trackers from on every inbound message and which
  keeps the entry alive through ``SessionMap.prune``; and ``memory_mode`` in the
  session's own transcript header, the field every memory reader already
  honours (``is_incognito_transcript``), so a transcript read never depends on
  the map being loaded. Omit it and the mode is in-memory only, which is also
  what a test double passing no ``sessions`` gets. The header write goes through
  the default ``ConversationLog``: every production log reads the one configured
  sessions directory, so it reaches the same file the channel writes.
* ``notify`` — an awaitable that delivers one confirmation message on the
  channel. The text is :data:`NOTICE_TEMPORARY` / :data:`NOTICE_INCOGNITO`, held
  here so two channels cannot describe the same mode differently.
* ``on_applied`` — optional, awaited once per NEWLY applied mode for the
  bookkeeping a mode change implies on that channel. Slack registers the thread
  (``set_slack_link``) so follow-up messages pass its in-active-thread gate; a
  channel that routes off the conversation id has nothing to do here.

Two entry points, because the two shapes are genuinely different:

* :func:`strip_and_apply` — one text in, ``(stripped_text, only_modifier)`` out.
  This is what a channel whose inbound message is a single string calls.
* :func:`strip_token` + :func:`apply_mode` — the primitives. Slack drives these
  directly because it carries TWO texts (the LLM-facing message and the
  mention-stripped command text) and only the command text decides whether the
  message was nothing BUT a modifier.

Applying a mode is idempotent: a repeat ``!incognito`` in an already-incognito
session neither re-audits nor re-notifies, so a user cannot spam the channel by
repeating the token.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from kiro_crew.sel import sel

logger = logging.getLogger(__name__)

#: Cap on each tracker. Bounded so a long-running bot serving many conversations
#: cannot grow these without limit; eviction is least-recently-marked.
PRIVACY_LRU_MAX = 10_000

#: The two mode names. These are also the ``SessionMap`` flag names, so the
#: durable spelling and the in-memory spelling cannot drift.
MODE_TEMPORARY = "temporary"
MODE_INCOGNITO = "incognito"

#: Confirmation text, one copy per mode. A channel renders it with its own
#: markup dialect; the words are shared so the two channels cannot promise
#: different guarantees for the same mode.
NOTICE_TEMPORARY = "🔒 Temporary mode ON — this thread won't read or save memory."
NOTICE_INCOGNITO = "🕶️ Incognito mode ON — this thread can read memory but won't save anything."

#: The refusal text, one per reason, for a modifier the gateway could not honour.
#: Both say the same three things in the same order: the mode was NOT applied,
#: the message was NOT processed (nothing ran, nothing was saved), and what the
#: user can do. Refusing is the fail-closed answer: running the message with the
#: mode silently dropped would be the leak the modifier exists to prevent.
NOTICE_REFUSED_LIMIT = (
    "🔒 Private-conversation limit reached ({cap} conversations are already kept "
    "private on this gateway): this one could not be made {mode}, so your message "
    "was NOT processed — nothing ran and nothing was saved. Send it again without "
    "the modifier to continue with memory on."
)
NOTICE_REFUSED_KEY = (
    "🔒 This conversation's identifier is too long for a private-conversation "
    "record, so it could not be made {mode} and your message was NOT processed — "
    "nothing ran and nothing was saved. Send it again without the modifier to "
    "continue with memory on."
)
NOTICE_REFUSED_PERSIST = (
    "🔒 The private-conversation record for this conversation could not be written, "
    "so it could not be made {mode} and your message was NOT processed — nothing "
    "ran and nothing was saved. Try again in a moment, or send it without the "
    "modifier to continue with memory on."
)
#: The refusal reasons, spelled as ``SessionMap.set_flag`` reports them -- plus
#: ``persist_failed``, which only a caller that REQUIRES the durable row
#: (``apply_mode(must_persist=True)``) ever sees.
REFUSAL_LIMIT = "limit"
REFUSAL_KEY_TOO_LONG = "key_too_long"
REFUSAL_PERSIST_FAILED = "persist_failed"


class PrivacyModeRefused(RuntimeError):
    """``apply_mode`` could not put the session into *mode*; the turn must not run.

    Raised only AFTER the refusal was audited (one SEL ``denied`` record) and the
    user told (``notify``), so a caller that merely lets it propagate has still
    refused fail-closed: no mark, no row, no turn, and the user knows why. The
    channel callers catch it to return quietly instead of logging an error.
    """

    def __init__(self, mode: str, session_key: str, reason: str) -> None:
        super().__init__(f"{mode} refused for {session_key[:80]!r}: {reason}")
        self.mode = mode
        self.session_key = session_key
        self.reason = reason


#: Standalone-token matchers. ``(?<!\S)`` / ``(?!\S)`` keep ``!incognito`` inside
#: a longer word (or a path) from matching, so only a token a user typed on its
#: own is a modifier.
TEMPORARY_TOKEN_RE = re.compile(r"(?<!\S)!temporary(?!\S)", re.IGNORECASE)
INCOGNITO_TOKEN_RE = re.compile(r"(?<!\S)!incognito(?!\S)", re.IGNORECASE)

#: Ordered ``(mode, pattern)`` pairs. Order is load-bearing: a caller that stops
#: at the first mode leaving nothing behind must check temporary first, matching
#: the shipped Slack ordering.
_MODES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (MODE_TEMPORARY, TEMPORARY_TOKEN_RE),
    (MODE_INCOGNITO, INCOGNITO_TOKEN_RE),
)

_PATTERNS: dict[str, "re.Pattern[str]"] = dict(_MODES)

#: Modes strictest-first, which is what :func:`strictest` ranks on. Declared
#: separately from :data:`_MODES` even though the two currently agree: that order
#: is about which token to STRIP first, and a third mode could need one position
#: for parsing and another for strength. Temporary leads because it forbids a
#: superset -- no reads, no writes, no persistence -- of what incognito forbids.
#: ``test_messaging_privacy_mode`` pins that every mode appears here, so a new one
#: has to state its rank rather than inherit a silent last place.
_STRICTNESS: tuple[str, ...] = (MODE_TEMPORARY, MODE_INCOGNITO)

#: session_key -> None. Values carry nothing; ``OrderedDict`` is here for the
#: LRU eviction order, not for a payload.
_temporary: "OrderedDict[str, None]" = OrderedDict()
_incognito: "OrderedDict[str, None]" = OrderedDict()

#: An awaitable that posts one line on the channel.
NoticeSender = Callable[[str], Awaitable[None]]
#: An awaitable run once per newly applied mode, given the mode name.
ModeHook = Callable[[str], Awaitable[None]]


def _tracker(mode: str) -> "OrderedDict[str, None]":
    """Return the LRU backing *mode*.

    Raises ``ValueError`` on an unknown mode rather than defaulting to one of
    them: a typo that silently marked the wrong mode would fail toward the
    permissive answer (incognito still reads memory; temporary does not).
    """
    if mode == MODE_TEMPORARY:
        return _temporary
    if mode == MODE_INCOGNITO:
        return _incognito
    raise ValueError(f"unknown privacy mode: {mode!r}")


def notice(mode: str) -> str:
    """Confirmation text for *mode*.

    Public because a channel needs it for the IDEMPOTENT case too:
    :func:`apply_mode` deliberately says nothing when the session is already
    marked, and a command that answers with silence reads as having failed.
    """
    return NOTICE_TEMPORARY if mode == MODE_TEMPORARY else NOTICE_INCOGNITO


def refusal_notice(mode: str, reason: str) -> str:
    """The text a channel posts when *mode* was refused for *reason*."""
    if reason == REFUSAL_KEY_TOO_LONG:
        return NOTICE_REFUSED_KEY.format(mode=mode)
    if reason == REFUSAL_PERSIST_FAILED:
        return NOTICE_REFUSED_PERSIST.format(mode=mode)
    from kiro_crew.session_map import PRIVACY_ROW_CAP

    return NOTICE_REFUSED_LIMIT.format(mode=mode, cap=PRIVACY_ROW_CAP)


#: Retained spelling for the module's own call sites.
_notice = notice


def mark(mode: str, session_key: str) -> None:
    """Record *session_key* in *mode*'s bounded LRU."""
    tracker = _tracker(mode)
    tracker[session_key] = None
    tracker.move_to_end(session_key)
    if len(tracker) > PRIVACY_LRU_MAX:
        tracker.popitem(last=False)


def mark_temporary(session_key: str) -> None:
    """Record *session_key* as temporary (blank-slate) in this process."""
    mark(MODE_TEMPORARY, session_key)


def mark_incognito(session_key: str) -> None:
    """Record *session_key* as incognito in this process."""
    mark(MODE_INCOGNITO, session_key)


def is_temporary(session_key: str) -> bool:
    """Whether *session_key* is in temporary mode (blocks memory READS too)."""
    return session_key in _temporary


def is_incognito(session_key: str) -> bool:
    """Whether *session_key* is in incognito mode (reads allowed, writes not)."""
    return session_key in _incognito


def is_restricted(session_key: str) -> bool:
    """Whether *session_key* must skip memory WRITES and persistence.

    The single predicate every enforcement site consults. Namespace-agnostic by
    construction — the key is only ever a dict lookup, so a Telegram, Discord or
    Slack key all answer the same way.
    """
    return session_key in _temporary or session_key in _incognito


def reset() -> None:
    """Drop every tracked session and pending reservation. For tests and gateway teardown only."""
    _temporary.clear()
    _incognito.clear()
    _pending.clear()


def conv_state_map(sessions: object) -> Any:
    """Return the ``SessionManager``'s canonical ``SessionMap``, or ``None``.

    The durable ``temporary``/``incognito`` flags are persisted through the SAME
    ``SessionMap`` instance the ``SessionManager`` owns, so writes stay
    consistent and no second instance can clobber them on save.

    The ``isinstance`` check is load-bearing, not defensive politeness: a bare
    ``getattr`` is satisfied by any attribute, and an auto-attribute stub (a
    ``MagicMock``) yields a stand-in whose ``get_flag`` returns a **truthy mock**
    for every flag. Readers would then mark every session both temporary and
    incognito — failing closed, but wrongly, and silently. Requiring the real
    class is what actually delivers the "test doubles fall back to in-memory
    only" contract.

    The import is deferred to call time on purpose: ``session_map`` imports
    ``messaging.link``, so a module-level import here would add a second edge
    into a package this one already sits inside.
    """
    from kiro_crew.session_map import SessionMap

    sm = getattr(sessions, "_session_map", None)
    return sm if isinstance(sm, SessionMap) else None


def hydrate(sessions: object, session_key: str) -> None:
    """Restore persisted privacy flags for *session_key* into the LRUs.

    Called once per session on the inbound path so a conversation marked
    temporary or incognito stays so across a gateway restart, when the
    process-local LRUs start empty. Idempotent and allocation-free for an
    unflagged key, so a caller may run it on every decision point.
    """
    sm = conv_state_map(sessions)
    if sm is None:
        return
    for mode, _pattern in _MODES:
        if sm.get_flag(session_key, mode):
            mark(mode, session_key)


def recorded_mode(sessions: object, session_key: str) -> str | None:
    """The strictest mode the trackers OR the durable map record for *session_key*.

    The read-only, any-thread form of :func:`hydrate` followed by
    :func:`is_temporary` / :func:`is_incognito`: the same two records, the same
    strictest-first answer, dict lookups only -- and nothing is marked. The
    trackers are mutated on the event loop (``mark`` is a three-step LRU
    update); a reader on a worker thread must not join in, so this reads the
    map flag beside the tracker instead of copying it in. ``None`` when neither
    record names a mode. The consolidator's write gate asks this before every
    store mutation, from whatever thread performs it.
    """
    sm = conv_state_map(sessions)
    for mode in _STRICTNESS:
        if session_key in _tracker(mode) or (sm is not None and sm.get_flag(session_key, mode)):
            return mode
    return None


def _persist(sessions: object, session_key: str, mode: str, *, strict: bool = False) -> None:
    """Write *mode*'s durable flag -- best-effort, unless *strict*.

    Best-effort because the in-memory mark follows regardless: a session whose
    flag could not be written is still restricted for the life of this process,
    which is the safe direction. Failing the modifier outright would leave the
    user believing the mode is off when it is on.

    *strict* is for a caller that is about to take an irreversible step ON THE
    STRENGTH of the row -- the Telegram steer path, whose steered message runs
    inside a turn that persists its transcript: there the row must exist before
    the step, so an I/O failure is a refusal (``persist_failed``), not a mark
    that holds for this process only.

    A :class:`~kiro_crew.session_map.PrivacyRowRefused` is never swallowed in
    either form: the map has declined to retain the row (its cap, or an
    over-long key), and :func:`apply_mode` must refuse the turn on it rather
    than leave a mark with no durable record behind it.
    """
    sm = conv_state_map(sessions)
    if sm is None:
        return
    # Call-time import for the same reason as ``conv_state_map``'s.
    from kiro_crew.session_map import PrivacyRowRefused

    try:
        sm.set_flag(session_key, mode, True)
    except PrivacyRowRefused:
        raise
    except Exception as exc:
        if strict:
            raise PrivacyRowRefused(session_key, REFUSAL_PERSIST_FAILED) from exc
        logger.warning(
            "could not persist %s mode for %s; it holds for this process only",
            mode,
            session_key,
            exc_info=True,
        )


async def refuse(
    mode: str,
    session_key: str,
    reason: str,
    *,
    source: str,
    caller: str = "system",
    resources: str = "",
    notify: NoticeSender | None = None,
) -> None:
    """Record and announce that *mode* was refused for *session_key* -- nothing else.

    The two side effects every refusal owes, in this order: the SEL record
    (``<source>.<mode>_mode`` / ``denied`` / ``private_session_refused:<reason>``
    plus the target, the denied twin of the record :func:`apply_mode` writes when
    it applies) and the user-facing notice. :func:`apply_mode` calls this when
    the map refuses the row.
    """
    sel().log_api_access(
        caller=caller,
        operation=f"{source}.{mode}_mode",
        outcome="denied",
        source=source,
        resources=f"private_session_refused:{reason}:{resources or session_key}",
    )
    if notify is not None:
        await notify(refusal_notice(mode, reason))


async def _persist_transcript_mode(session_key: str, mode: str) -> None:
    """Record *mode* as the transcript header's ``memory_mode``, best-effort.

    The record the memory readers consult. The session map flag is what the
    channel's own gate hydrates from and what keeps the map entry alive through
    ``SessionMap.prune``; the transcript header is the field every memory reader
    already refuses on (``is_incognito_transcript``, the consolidator's header
    source, the dashboard's persisted probe, the consolidate route's own header
    probe), and it lives WITH the Kiro Crew transcript -- the turns written
    before the modifier -- so it travels with what it protects and is reclaimed
    with it and not before. Two records rather than one because they answer two
    readers: a transcript read must not depend on the map being loaded, and the
    channel gate must not pay a transcript read per inbound message.

    Upserted through ``ConversationLog.update_metadata_if``: a transcript that
    does not exist yet gets a metadata-only line, and ``ConversationLog.append``
    keeps an existing header, so the first row any later writer appends lands
    under a header that already carries the mode. Tighten-only: the guard
    admits the write only when *mode* is at least as strict as the header's
    current mode, so ``!incognito`` typed after ``!temporary`` cannot re-enable
    memory reads, and a header already carrying *mode* is rewritten unchanged.

    The default ``ConversationLog`` reaches the same file the channel writes:
    every production log reads the one configured sessions directory. Off the
    loop, because ``update_metadata_if`` takes the transcript's cross-process
    flock. Best-effort for the same reason ``_persist`` is: the in-memory mark
    has already happened, so a failed header write leaves the session restricted
    for this process and is logged, not raised.
    """
    # Call-time import because the cycle is real: ``kiro_crew.history`` imports
    # ``history_consolidation`` at module scope (facade re-exports), and that
    # module imports THIS one at module scope for the session-map source of a
    # consolidation target's mode -- so a module-scope import here would load
    # ``history`` while ``history`` is still loading. Deferred, the edge is paid
    # once, at the first mark.
    from kiro_crew.history import ConversationLog, transcript_privacy_mode

    log = ConversationLog()

    def _tighten_only(metadata: dict) -> bool:
        # Normalized first: a raw ``Temporary`` would compare as unknown, and an
        # incognito stamp would then overwrite the stricter mode. No write at
        # all when the header already records the mode.
        return needs_tightening(transcript_privacy_mode(metadata.get("memory_mode")), mode)

    try:
        await asyncio.to_thread(
            log.update_metadata_if, session_key, {"memory_mode": mode}, _tighten_only
        )
    except Exception:
        logger.warning(
            "could not record %s mode in the transcript header of %s; "
            "the session map flag holds it",
            mode,
            session_key,
            exc_info=True,
        )


def strip_token(text: str, mode: str) -> tuple[str, bool]:
    """Remove *mode*'s standalone token from *text*.

    Returns ``(cleaned_text, found)``. When the token is absent *text* is handed
    back untouched — no whitespace collapse — so a caller can tell "nothing to do
    here" from "stripped down to nothing".
    """
    pattern = _PATTERNS.get(mode)
    if pattern is None:
        raise ValueError(f"unknown privacy mode: {mode!r}")
    new, n = pattern.subn("", text)
    if not n:
        return text, False
    return " ".join(new.split()), True


def strip_tokens(text: str) -> tuple[str, tuple[str, ...]]:
    """Strip every modifier token from *text*.

    Returns ``(cleaned_text, modes_found)`` with *modes_found* in
    :data:`_MODES` order. Applies nothing; this is the pure half, for a caller
    that needs the cleaned text without the side effects (a log line, a preview).
    """
    found: list[str] = []
    for mode, _pattern in _MODES:
        text, had = strip_token(text, mode)
        if had:
            found.append(mode)
    return text, tuple(found)


def strictest(modes: "list[str] | tuple[str, ...]") -> str:
    """The strictest of *modes*, or ``""`` for none.

    For a caller that has to collapse several requests into ONE decision: a queue
    drain answers a burst of messages as a single turn under a single key, so it can
    honour exactly one mode. Taking the strictest is the only choice that cannot
    silently downgrade what a user asked for -- honouring the first would let a later
    request in the same burst be dropped, and honouring the last would drop an
    earlier one.

    Unknown names are ignored rather than raising: this runs on the delivery path,
    where refusing a turn over an unrecognized mode would cost the user their message
    to protect them from a mode that does not exist.
    """
    present = set(modes)
    return next((mode for mode in _STRICTNESS if mode in present), "")


def needs_tightening(current: str, mode: str) -> bool:
    """Whether a record holding *current* must be rewritten to hold *mode*.

    True only when *mode* is STRICTER than *current*; an absent or unknown
    current (``""``) counts as weakest. A record already at *mode* answers
    False -- so a tighten-only writer that asks this performs NO write for a
    header that already records the mode (the startup stamp would otherwise
    rewrite every flagged row's header on every boot) -- and a stricter record
    answers False too, which is the tighten-only rule itself.
    """
    return current != mode and strictest([current, mode]) == mode


async def apply_mode(
    mode: str,
    session_key: str,
    *,
    source: str,
    caller: str = "system",
    resources: str = "",
    sessions: object | None = None,
    notify: NoticeSender | None = None,
    on_applied: ModeHook | None = None,
    must_persist: bool = False,
) -> bool:
    """Put *session_key* into *mode*, and tell the user once.

    Returns whether the mode was NEWLY applied. Idempotent: an already-marked
    session returns ``False`` without re-auditing or re-notifying, so repeating
    the token costs the channel nothing.

    Raises :class:`PrivacyModeRefused` when the session map declines to retain
    the row (``PRIVACY_ROW_CAP`` reached, or an over-long key): the refusal is
    audited (one SEL ``denied`` record) and the user told through *notify*
    BEFORE the raise, nothing is marked or written, and the caller must not run
    the turn -- the message is refused, never run with the mode dropped. With
    *must_persist* the row is DURABLE before this returns -- a write the map
    cannot perform (``persist_failed``, at ``set_flag`` or at the awaited
    ``aflush``) refuses the same way instead of falling back to a
    process-local mark, and a flush that fails takes the mark back with it: for
    a caller that is about to take an irreversible step on the strength of the
    row (the Telegram steer path reserves the row this way BEFORE it steers).

    Ordering is deliberate. The durable row is admitted first, because it is
    what can refuse, then the in-memory mark lands -- both synchronous, so
    before any await: a concurrent inbound message on the same session cannot
    observe the session as unrestricted after the user asked for privacy, and
    two modifiers racing for the last row cannot both take it. With
    *must_persist* the map's write is awaited right after the mark, under the
    mark's cover. The
    audit follows, still before any other await, so a task cancelled while the
    header write below is in flight (a gateway shutdown right after the
    modifier) has already written the audit record; the idempotency return
    above would never come back to write it. Then the awaited transcript-header
    write, the caller's hook and the notice.
    """
    if session_key in _tracker(mode):
        return False
    # Call-time import for the same reason as ``conv_state_map``'s.
    from kiro_crew.session_map import PrivacyRowRefused

    if sessions is not None:
        try:
            _persist(sessions, session_key, mode, strict=must_persist)
        except PrivacyRowRefused as refused:
            await refuse(
                mode,
                session_key,
                refused.reason,
                source=source,
                caller=caller,
                resources=resources,
                notify=notify,
            )
            raise PrivacyModeRefused(mode, session_key, refused.reason) from refused
    mark(mode, session_key)
    sel().log_api_access(
        caller=caller,
        operation=f"{source}.{mode}_mode",
        outcome="allowed",
        source=source,
        resources=resources or session_key,
    )
    if sessions is not None and (sm := conv_state_map(sessions)) is not None:
        # The row set above is in the map's memory with a DEBOUNCED write owed;
        # it is awaited here so nothing published past this point -- the header
        # stamp, the hook, the notice, the return -- claims a row that has not
        # landed. After the mark, not before: the mark is what keeps a concurrent
        # message on this session from observing it unrestricted while the disk
        # write runs. On failure the two forms part: ``must_persist`` takes the
        # mark back and refuses (the caller's irreversible step must not run on
        # a row that is not durable); the best-effort form keeps the mark -- the
        # session stays restricted for this process, the safe direction -- and
        # says so in the log, since telling the user the mode is off while it
        # is on would be the worse error. The audit above is written before this
        # await, so a task cancelled mid-write has already recorded the mode.
        try:
            await sm.aflush()
        except Exception as exc:
            if not must_persist:
                logger.warning(
                    "%s mode for %s is marked but its map row did not reach disk; "
                    "it holds for this process, the map's next write retries",
                    mode,
                    session_key,
                    exc_info=True,
                )
            else:
                _tracker(mode).pop(session_key, None)
                try:
                    sm.set_flag(session_key, mode, False)
                except Exception:
                    logger.warning(
                        "could not clear the %s flag of %s after a failed flush", mode, session_key
                    )
                await refuse(
                    mode,
                    session_key,
                    REFUSAL_PERSIST_FAILED,
                    source=source,
                    caller=caller,
                    resources=resources,
                    notify=notify,
                )
                raise PrivacyModeRefused(mode, session_key, REFUSAL_PERSIST_FAILED) from exc
    if sessions is not None:
        await _persist_transcript_mode(session_key, mode)
    if on_applied is not None:
        await on_applied(mode)
    if notify is not None:
        await notify(_notice(mode))
    return True


@dataclass(frozen=True)
class Reservation:
    """A mode applied ahead of an irreversible step, so it can be taken back if the step fails.

    Returned by :func:`reserve`; ended by exactly one :func:`commit` (the step
    happened) or :func:`release` (it did not). What a release may take back is
    decided by the pending state the holders of one (mode, key) share
    (:class:`_Pending`), not by the holder releasing.
    """

    mode: str
    session_key: str


@dataclass
class _Pending:
    """The shared state of one (mode, key) reservation while any holder is pending.

    Registered by the FIRST holder before its first await, so a second caller
    for the same (mode, key) always finds it -- there is no window in which two
    callers each believe they hold the only reservation and the loser's release
    erases the winner's committed mode. *settled* is set once the first
    holder's application has finished, one way or the other: a joiner waits on
    it, then rides the application (*applied*, *header_before* are the group's)
    or, if it *failed*, attempts its own. *committed*: a holder's step landed,
    so the mode is the conversation's for good, whatever the other holders'
    steps do.
    """

    settled: asyncio.Event
    holders: int = 0
    applied: bool = False
    header_before: object = None
    committed: bool = False
    failed: bool = False


#: (mode, session_key) -> the shared reservation state while any holder is pending.
#: Refcounted because two modifiers on the same thread can both reserve before
#: either step lands: the second finds the group already registered and rides
#: it, so the first holder's failed step must not take the mode away from under
#: the second's message. Only the last release of an uncommitted group that
#: applied the mode loosens anything.
_pending: dict[tuple[str, str], _Pending] = {}


async def reserve(
    mode: str,
    session_key: str,
    *,
    source: str,
    caller: str = "system",
    resources: str = "",
    sessions: object | None = None,
    notify: NoticeSender | None = None,
    on_applied: ModeHook | None = None,
) -> Reservation:
    """Apply *mode* strictly, ahead of a step that cannot be taken back.

    :func:`apply_mode` with ``must_persist=True`` -- row durable, mark, audit,
    header, hook and notice, or :class:`PrivacyModeRefused` -- plus the
    bookkeeping a later :func:`release` needs. The Telegram steer path reserves
    before it steers: the message it is about to fold into a running turn must
    already be protected by the row that turn's transcript write and the
    channel gate read, and a row that cannot be taken or made durable must
    refuse the message before it is in the turn. Every reservation must end in
    exactly one :func:`commit` or :func:`release`.

    The first caller for a (mode, key) registers the shared holder BEFORE its
    first await and runs the application; a concurrent second caller joins the
    registered holder and waits for that application to settle instead of
    running its own, so the two cannot race each other's bookkeeping. If the
    first application fails, its holder is unwound and the joiner attempts the
    application itself, as the new first holder.
    """
    key = (mode, session_key)
    pending = _pending.get(key)
    if pending is not None:
        pending.holders += 1
        await pending.settled.wait()
        if not pending.failed:
            return Reservation(mode, session_key)
        # The application this call would have ridden was refused or raised, and
        # its holder is gone: attempt one of our own.
        pending.holders -= 1
        return await reserve(
            mode,
            session_key,
            source=source,
            caller=caller,
            resources=resources,
            sessions=sessions,
            notify=notify,
            on_applied=on_applied,
        )
    pending = _Pending(settled=asyncio.Event(), holders=1)
    _pending[key] = pending
    try:
        header_before: object = None
        if sessions is not None and session_key not in _tracker(mode):
            from kiro_crew.history import ConversationLog

            try:
                header_before = (
                    await asyncio.to_thread(ConversationLog().get_metadata, session_key)
                ).get("memory_mode")
            except Exception:
                logger.debug("could not read %s's header before reserving %s", session_key, mode)
        applied = await apply_mode(
            mode,
            session_key,
            source=source,
            caller=caller,
            resources=resources,
            sessions=sessions,
            notify=notify,
            on_applied=on_applied,
            must_persist=True,
        )
    except BaseException:
        # Unwind: this holder leaves, and the joiners waiting on it learn the
        # application failed. The entry is retired here whatever their count --
        # they hold the object and read ``failed`` off it.
        pending.failed = True
        pending.holders -= 1
        if _pending.get(key) is pending:
            _pending.pop(key, None)
        pending.settled.set()
        raise
    if applied:
        pending.applied = True
        pending.header_before = header_before
    pending.settled.set()
    return Reservation(mode, session_key)


def commit(reservation: Reservation) -> None:
    """The irreversible step happened: the mode is the conversation's for good.

    Every holder of the same (mode, key) is committed with it -- a later release
    by a holder whose own step failed loosens nothing, because a message did
    land under the mode.
    """
    key = (reservation.mode, reservation.session_key)
    pending = _pending.get(key)
    if pending is None:
        return
    pending.committed = True
    pending.holders -= 1
    if pending.holders <= 0:
        _pending.pop(key, None)


async def release(
    reservation: Reservation,
    *,
    sessions: object | None,
    source: str,
    caller: str = "system",
    resources: str = "",
) -> bool:
    """The irreversible step did NOT happen: take back what the reservation applied.

    Loosening a privacy mode is otherwise never done, so the conditions are
    narrow and each one is checked: the mode is taken back only when this is
    the LAST pending holder, no holder committed, and the group newly applied
    the mode -- then the map flag is cleared AND its write awaited (the row
    stops counting against the cap; a row that carried another privacy flag
    keeps that one), the transcript header is restored to what it recorded
    before, only if it still records what the stamp wrote, and only then is
    the tracker mark dropped and the reversal reported: one SEL record
    (``released``) beside the ``allowed`` record the reservation wrote.
    Durable first, published last -- the mirror of ``apply_mode``'s order. If
    the flag's write fails, the mode is RETAINED, fail-closed toward private:
    the flag is put back in the map's memory (the map's next write retries),
    the mark stays, the header is left as stamped, and the failure is reported
    as one ``retained`` record instead of ``released``. Returns whether the
    mode was released. A session that was already in the mode is left exactly
    as it was.
    """
    key = (reservation.mode, reservation.session_key)
    pending = _pending.get(key)
    if pending is None:
        return False
    pending.holders -= 1
    if pending.holders > 0:
        return False
    _pending.pop(key, None)
    if pending.committed or not pending.applied:
        return False
    mode, session_key = reservation.mode, reservation.session_key
    sm = conv_state_map(sessions)
    if sm is not None:
        try:
            sm.set_flag(session_key, mode, False)
            await sm.aflush()
        except Exception:
            # The clear did not reach disk: retain the mode. The row goes back
            # into the map's memory so the state the next write lands is the
            # private one, the mark never left, the header still says the mode.
            logger.warning(
                "%s mode of %s could not be released durably; the mode is retained",
                mode,
                session_key,
                exc_info=True,
            )
            try:
                sm.set_flag(session_key, mode, True)
            except Exception:
                logger.warning("could not restore the %s flag of %s", mode, session_key)
            sel().log_api_access(
                caller=caller,
                operation=f"{source}.{mode}_mode",
                outcome="retained",
                source=source,
                resources=f"release_failed:{REFUSAL_PERSIST_FAILED}:{resources or session_key}",
            )
            return False
    if sessions is not None:
        from kiro_crew.history import ConversationLog, transcript_privacy_mode

        restored = pending.header_before if pending.header_before is not None else "persistent"

        def _still_ours(metadata: dict) -> bool:
            return transcript_privacy_mode(metadata.get("memory_mode")) == mode

        try:
            await asyncio.to_thread(
                ConversationLog().update_metadata_if,
                session_key,
                {"memory_mode": restored},
                _still_ours,
                require_existing=True,
            )
        except Exception:
            # The header is the stricter record now; every memory reader refuses
            # on it, the safe direction. Logged, not fatal: the durable flag is
            # already cleared, so the mark follows the flag.
            logger.warning("could not restore %s's header on release of %s", session_key, mode)
    _tracker(mode).pop(session_key, None)
    sel().log_api_access(
        caller=caller,
        operation=f"{source}.{mode}_mode",
        outcome="released",
        source=source,
        resources=resources or session_key,
    )
    return True


async def strip_and_apply(
    text: str,
    session_key: str,
    *,
    source: str,
    caller: str = "system",
    resources: str = "",
    sessions: object | None = None,
    notify: NoticeSender | None = None,
    on_applied: ModeHook | None = None,
) -> tuple[str, bool]:
    """Strip the privacy modifiers from *text* and apply each one found.

    Returns ``(stripped_text, only_modifier)``:

    * *stripped_text* — *text* with every modifier token removed and whitespace
      collapsed. The token MUST NOT reach the model: it is an instruction to the
      gateway, and a prompt containing it invites the model to answer it.
    * *only_modifier* — ``True`` when the message was nothing but modifier(s).
      The caller MUST then return without starting a turn; there is no question
      to answer, and running one would spend a turn on the word ``!incognito``.

    Modes are applied in :data:`_MODES` order, stopping as soon as nothing is
    left to say — so ``!temporary`` alone applies temporary and returns, exactly
    as the shipped Slack path does. A :class:`PrivacyModeRefused` from
    :func:`apply_mode` propagates: the user has been told, and the caller must
    not run the turn.
    """
    for mode, _pattern in _MODES:
        stripped, had = strip_token(text, mode)
        if not had:
            continue
        await apply_mode(
            mode,
            session_key,
            source=source,
            caller=caller,
            resources=resources,
            sessions=sessions,
            notify=notify,
            on_applied=on_applied,
        )
        text = stripped
        if not text:
            return text, True
    return text, False
