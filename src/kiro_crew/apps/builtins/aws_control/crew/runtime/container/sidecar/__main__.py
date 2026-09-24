"""The backup process: a cycle every interval, and one more on the way out.

Run as ``python -m container.sidecar``. The supervisor starts it after the front, and
only when a bucket is configured: with no bucket there is nothing to write to, which is
a crew running without durability rather than a fault.

## A failed cycle does not end the process

This is the one place where the obvious posture is the wrong one. A process that exits
on a failed upload would be reported to the supervisor as a dead child, the task would
be torn down, and the replacement would come up without the turns the cycle had not
yet written -- so the response to "the backup failed" would be to destroy the data the
backup exists to protect.

So a cycle that fails is logged at ERROR and the loop continues. The next cycle
re-uploads everything still unrecorded, because the fingerprint map only remembers
uploads that SUCCEEDED. A throttled bucket or a brief credential gap costs latency on
the durability window, not the window's contents.

## A bucket that cannot be written at all DOES end the process

The exception is a failure no retry resolves: a denied ``PutObject``, a bucket name
that does not exist, a credential that is not valid. Retrying those is worse than
exiting, because the log fills with attempts while the task keeps taking turns that
nothing will ever save -- the appearance of durability, which is the one failure this
pair exists to remove. The store classifies them as ``StoreUnusable`` and this process
ends on it, so the fault is reported instead of accumulating silently.

The process also exits non-zero if it cannot work at all -- no bucket, unreadable
configuration -- for the same reason: those are deployment mistakes that will not
resolve by retrying, and a task whose sidecar is silently absent has the appearance of
durability. The supervisor treats this child's death as fatal because a crew that keeps
serving with no writer accumulates an unbounded amount of state nothing will save, and
losing at most one interval loudly is better than losing everything quietly.

## The final cycle

SIGTERM interrupts the wait and the loop runs one more cycle -- one that BEGINS after
the signal was observed, and whose completion the process waits for before returning.
That ordering is the whole point. A cycle already in flight when the signal arrives
started before the backend's flush, so it cannot contain what that flush produced;
accepting it as the final one would lose exactly the turns the final cycle exists to
save. On an orderly replacement -- the common case, since a deploy is one -- the
supervisor drains the front so no new turn arrives, then the backend so it flushes what
it holds, and only then this process, whose last act is to upload what that flush
wrote. A post-stop cycle that does not complete exits non-zero rather than reporting a
clean stop.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Sequence

from .. import common
from ..common import Settings
from . import backup as backup_mod
from .store import ObjectStore, S3ObjectStore, StoreUnusable

log = logging.getLogger("container.sidecar")

#: Set by SIGTERM/SIGINT. Interrupts the wait between cycles.
_STOP = threading.Event()


def _on_signal(signum, _frame) -> None:
    log.info("sidecar: signal %s; one final cycle then exit", signum)
    _STOP.set()


def run(
    settings: Settings,
    store: ObjectStore,
    *,
    stop: threading.Event | None = None,
    max_cycles: int | None = None,
) -> int:
    """Back up every interval until stopped, then once more. Return an exit code.

    ``stop`` and ``max_cycles`` are injected so a test can drive real cycles without
    signals and without waiting: ``max_cycles`` bounds the loop, and a test that sets
    ``stop`` before the first wait gets exactly the shutdown path.
    """
    stopping = _STOP if stop is None else stop
    state: dict[str, backup_mod.Fingerprint] = {}
    cycles = 0
    consecutive_failures = 0

    while True:
        # Read BEFORE the cycle runs. A cycle already in flight when the signal lands
        # started before the backend's flush, so its uploads cannot contain what that
        # flush produced -- accepting it as the final one is how an orderly replacement
        # silently loses its last turns. This makes the final cycle one that BEGINS
        # after the stop was observed, and the loop returns only once it has finished.
        final = stopping.is_set()
        cycles += 1
        completed = _one_cycle(settings, store, state, attempt=consecutive_failures + 1)
        consecutive_failures = 0 if completed else consecutive_failures + 1
        if final:
            if completed:
                log.info("sidecar: post-shutdown cycle complete; stopped after %d cycle(s)", cycles)
                return 0
            log.error(
                "sidecar: the post-shutdown cycle did not complete, so state written "
                "after the backend's flush may not be in the bucket. Exiting non-zero "
                "so the replacement's operator sees it."
            )
            return 1
        if max_cycles is not None and cycles >= max_cycles:
            return 0
        # Woken by the signal rather than by the interval means the next pass is the
        # final cycle: it reads the flag as set, runs whole, and returns.
        stopping.wait(settings.backup_interval_secs)


def _one_cycle(
    settings: Settings,
    store: ObjectStore,
    state: dict[str, backup_mod.Fingerprint],
    *,
    attempt: int,
) -> bool:
    """Run one cycle. ``True`` if it completed, ``False`` if it was logged as failed.

    Failure is reported and not raised, because the loop must not end on it (see the
    module docstring). The one exception is :class:`StoreUnusable`, which is not a
    failed request but the bucket being unwritable: it leaves here so the process can
    end on it, because no later cycle gets a different answer.
    """
    try:
        backup_mod.run_cycle(settings, store, state=state)
    except StoreUnusable:
        raise
    except backup_mod.BackupIncomplete as exc:
        log.error(
            "sidecar: cycle incomplete (consecutive failure %d) -- %s. Retrying at the "
            "next interval; the objects that were uploaded are recorded and are not "
            "sent again.",
            attempt,
            exc,
        )
        return False
    except Exception:  # noqa: BLE001 - logged, never fatal to the loop
        log.exception(
            "sidecar: cycle failed (consecutive failure %d). Retrying at the next "
            "interval rather than exiting: a dead sidecar tears the task down, which "
            "would lose the state this process exists to save.",
            attempt,
        )
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = common.load()
    if not settings.backup_bucket:
        # The supervisor does not start this process without a bucket, so reaching here
        # means the image was launched some other way. Refused rather than idled: a
        # sidecar that runs and writes nothing is the appearance of durability.
        log.error(
            "sidecar: SMC_BACKUP_BUCKET is not set, so there is nowhere to write this "
            "task's state. Refusing to run rather than idling, which would look like a "
            "working backup."
        )
        return 2
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    log.info(
        "sidecar: backing up to s3://%s every %ds",
        settings.backup_bucket,
        settings.backup_interval_secs,
    )
    try:
        return run(settings, S3ObjectStore(settings.backup_bucket))
    except StoreUnusable as exc:
        # Not retried, because the answer does not change: a denied PutObject, a bucket
        # that does not exist, a credential that is not valid. Continuing to serve turns
        # while the writer cannot write any of them is the appearance of durability, so
        # the process ends and the supervisor reports it.
        log.error("sidecar: %s", exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
