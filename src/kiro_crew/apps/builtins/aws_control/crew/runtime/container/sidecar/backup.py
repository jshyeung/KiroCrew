"""One backup cycle: what must be durable, and how each object is copied.

## What is in the set

Three kinds, and the set is a definition rather than a filter, so "was this file
backed up" has an answer that does not depend on what the directory happened to
hold:

1. **The two authority files**, ``session_map.json`` and ``open_slots.json``. They
   turn a slot id back into a conversation, so without them the transcripts are on
   disk and the conversation list is empty.
2. **Every live transcript**, the ``.jsonl`` files directly under the sessions
   directory. One per conversation this task has served.
3. **Every archived segment** under ``sessions/archive/``. Rotation moves the older
   part of a long conversation there and the container never reads it back -- the
   front's fetch may not list, and finding a segment requires listing -- so these are
   uploaded for the owner's control plane, which has credentials of its own. Leaving
   them behind would be silent loss of the older half of every long conversation.

Anything else under the sessions directory is not in the set. The front writes a
temporary file there while fetching and unlinks it itself, and a name that is
neither that nor a transcript is not a conversation.

## What order they go in, and why the order is a correctness rule

The transcripts go first and the authority files last, in their own phase. The
authority files are the INDEX: a replacement reads them to decide which conversations
exist, and the front then fetches each named transcript lazily. So an authority table
newer than the transcripts it names points at objects that are not in the bucket, and
the front reads an absent transcript as a conversation with no history -- a live
conversation served empty, with nothing raised anywhere. The opposite skew is
harmless: an authority table older than the transcripts names only slots whose bytes
are already there, and a transcript it does not name yet is unreferenced rather than
misread.

Which is why the authority files are OPENED first, before a single transcript is
listed, and sent from those descriptors at the end. Opening fixes the instant a file
describes, so the pair is one coherent snapshot of the index taken before the
enumeration it indexes. Reading them at send time instead let a slot table flushed
during the cycle name a transcript that cycle never listed.

That rests on the backend publishing both files the way it publishes a transcript, a
temporary file and a rename, which leaves an open descriptor addressing the whole
previous version. It does: ``session_map.json`` and ``open_slots.json`` are both written
through an atomic replace. A writer that truncated one in place instead would take the
snapshot property away without changing anything here, so it is pinned by a test rather
than left as an assumption.

For the same reason the authority phase is SKIPPED entirely when the transcript phase
refused anything. Publishing it then would advance the index past bytes this cycle
failed to write; withholding it leaves the pair at the last cycle that completed,
which is older and coherent.

One residual remains in the pair itself. The two files are two PUTs, so a failure
between them leaves the bucket holding one from this cycle's snapshot and one from an
earlier cycle's. Both were opened before this cycle's enumeration, so neither names a
transcript that is absent, and the failure raises rather than passing quietly; the cost
is one interval in which the two files disagree about which slots exist, which the next
cycle resolves.

## How one object is copied

``open_snapshot`` opens the file ONCE and records the length that descriptor's file
had at that moment. The upload then sends exactly that many bytes from that
descriptor. Three properties follow, and they are the three constraints this design
has to hold at the same time:

* **Consistent** without a lock. The backend publishes a transcript with a temporary
  file and a rename, so it never writes into the bytes behind an open descriptor --
  it swaps the directory entry to a different inode. A descriptor opened before the
  swap keeps addressing a whole, finished version, and a file that is appended to
  instead is uploaded as the prefix that existed at open time, which is also a
  version that was really on disk.
* **Bounded** on disk. Nothing is copied first. A cycle spends one descriptor and one
  fixed transport buffer per object, so an oversized artifact cannot fill the
  filesystem the app is writing to.
* **Nothing dropped.** There is no size at which an object is skipped. An entry that
  genuinely cannot be uploaded is recorded and the cycle ends by RAISING
  :class:`BackupIncomplete` -- after uploading everything it could, so one bad entry
  does not cost every other conversation its backup.

## Every shape an entry can have, and what happens to it

| entry                                      | verdict                                |
| ------------------------------------------ | -------------------------------------- |
| regular file, one link                     | uploaded                               |
| regular file, several links                | uploaded: the descriptor still         |
|                                            | addresses real bytes, and this side     |
|                                            | only reads them                        |
| regular file that grew since it was opened  | uploaded to its length at open         |
| regular file that shrank since it was opened| uploaded short, and the declared length |
|                                            | makes the transport fail rather than    |
|                                            | pad; recorded, so the cycle raises      |
| zero bytes                                 | uploaded: an empty conversation is a    |
|                                            | conversation                            |
| above the reader's ceiling                 | uploaded, with a warning naming it:     |
|                                            | backed up, and the front will refuse to |
|                                            | restore it, so an operator hears it     |
|                                            | before a customer does                  |
| symlink                                    | recorded; the cycle raises              |
| reached through a symlinked directory      | recorded; the cycle raises, and a       |
|                                            | linked archive root is refused before    |
|                                            | anything under it is listed at all       |
| directory, FIFO or socket                  | recorded; the cycle raises              |
| gone between listing and opening           | counted as gone; the cycle continues,   |
|                                            | because a deleted conversation is not   |
|                                            | a backup failure                        |
| unchanged since its last upload            | not re-uploaded                         |

## What the fingerprint is for, and what it is not

Change detection is a COST decision, not a correctness one. The fingerprint is the
inode, the length and the modification time as they were at open, and an object is
re-uploaded whenever it differs from the one last uploaded successfully. It lives in
memory, so a restarted sidecar re-uploads everything once: paying for a full cycle is
the right way to be wrong here, and persisting the state would put a second authority
on disk to keep in agreement with the bucket.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from ..common import Settings, keys
from ..common.config import MAX_OBJECT_BYTES
from .store import ObjectStore, StoreUnusable

log = logging.getLogger("smc.sidecar.backup")

__all__ = [
    "Fingerprint",
    "Snapshot",
    "BackupSet",
    "CycleResult",
    "BackupIncomplete",
    "open_snapshot",
    "objects_to_back_up",
    "run_cycle",
]

#: Flags for opening a file to be uploaded.
#:
#: ``O_NOFOLLOW`` refuses a symlink at the final component, so a link planted where a
#: transcript belongs is reported instead of followed to whatever it points at.
#: ``O_NONBLOCK`` is what keeps the open from hanging: opening a FIFO for reading blocks
#: until a writer arrives, and an entry planted as a FIFO would otherwise stall the cycle
#: indefinitely rather than be refused.
_OPEN_FLAGS: int = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)

#: Flags for opening one DIRECTORY component on the way down to a file.
#:
#: ``O_NOFOLLOW`` is what makes the descent safe. ``O_NOFOLLOW`` on the file alone
#: guards only the last name, so a link planted at ``sessions/archive`` -- a directory
#: the agent writes in -- is descended normally and every regular file behind it opens
#: and uploads. Walking down with this flag at each step means a link ANYWHERE in the
#: chain is refused instead, so the only files that reach the bucket are files reached
#: through real directories inside the data home.
_DIR_FLAGS: int = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class Fingerprint:
    """What an object looked like when it was last uploaded successfully."""

    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class Snapshot:
    """An open descriptor and the length its file had when it was opened.

    The pair IS the snapshot. Neither half is a snapshot alone: the descriptor without
    the length would upload however much had arrived by the time the transport got
    there, and the length without the descriptor would have to re-open the name, which
    is a second resolution of one path with a window in between.
    """

    fh: BinaryIO
    fingerprint: Fingerprint

    @property
    def size(self) -> int:
        return self.fingerprint.size

    def close(self) -> None:
        self.fh.close()


class RefusedEntry(RuntimeError):
    """This entry is not a file whose bytes can be uploaded."""


def _descend(root: Path, parts: tuple[str, ...]) -> int:
    """Open the directory at *root* / *parts*, refusing a symlink at any component.

    Returns a descriptor the caller must close. ``ELOOP`` from any step means a
    directory in the chain is a link, which is refused rather than followed: a link out
    of the data home turns "back up this task's own state" into "upload whatever it
    points at", and the sessions tree is one the agent writes in.
    """
    fd = os.open(str(root), _DIR_FLAGS)
    try:
        for name in parts:
            nxt = os.open(name, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
    except OSError:
        os.close(fd)
        raise
    return fd


def _open_within(root: Path, path: Path) -> int:
    """Open *path* for reading, having walked to it from *root* one component at a time.

    *root* is the trust anchor -- the data home, which is the container's own mount and
    not a path the agent can replace. Every component below it is opened with the link
    refused, so the descriptor returned addresses a file inside the real data home and
    not one reached through a directory something swapped for a link.

    Opening the full path in one call cannot do this: ``O_NOFOLLOW`` applies to the last
    component only, and the kernel resolves the rest normally.
    """
    rel = path.relative_to(root)
    parts = rel.parts
    fd = _descend(root, parts[:-1])
    try:
        return os.open(parts[-1], _OPEN_FLAGS, dir_fd=fd)
    finally:
        os.close(fd)


@dataclass
class CycleResult:
    """What one cycle did, per object, for the log and for the tests."""

    uploaded: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)
    above_ceiling: list[str] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)
    #: Authority keys this cycle deliberately did not publish, because a transcript in
    #: the same cycle was refused. They stay as the last complete cycle left them.
    withheld: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.refused

    def summary(self) -> str:
        return (
            f"{len(self.uploaded)} uploaded, {len(self.unchanged)} unchanged, "
            f"{len(self.gone)} gone, {len(self.refused)} refused, "
            f"{len(self.withheld)} authority withheld"
        )


class BackupIncomplete(RuntimeError):
    """At least one object in the set could not be uploaded.

    Raised at the END of the cycle, with everything that could be uploaded already
    uploaded. The distinction matters: refusing the whole cycle on the first bad entry
    would cost every other conversation its backup, and dropping the bad entry with a
    log line would be the silent loss this design exists to prevent. So the cycle does
    all the work it can and then cannot be ignored.
    """

    def __init__(self, result: CycleResult) -> None:
        self.result = result
        detail = "; ".join(f"{name}: {why}" for name, why in result.refused)
        super().__init__(
            f"{len(result.refused)} object(s) in the backup set could not be uploaded "
            f"({detail}). Everything else in this cycle was uploaded."
        )


def open_snapshot(path: Path, *, root: Path) -> Snapshot | None:
    """Open *path* for upload, or ``None`` when it is not there any more.

    ``None`` means the file was listed and then removed, which is a conversation the
    owner deleted rather than a backup failure. Every other way this can fail RAISES
    :class:`RefusedEntry`, because those are entries that are present and are not
    files whose bytes belong in the bucket.

    Shape is decided on the DESCRIPTOR, never on the name: a check by name followed by
    an open by name is two resolutions of one path with a window in between. Opening
    first with the link refused and then reading ``fstat`` off the descriptor means the
    entry judged is exactly the entry that will be uploaded.

    *root* is the data home, and the open walks down to *path* from it one component at
    a time with each link refused, so an ancestor directory replaced by a link is a
    refusal here and not a file uploaded from outside the data home.
    """
    try:
        fd = _open_within(root, path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            # Both codes mean the same refusal. ``O_NOFOLLOW`` on a symlink reports
            # ELOOP for the final component and, combined with ``O_DIRECTORY``, ENOTDIR
            # for a directory component -- so the two are one case: something on the way
            # to this file is a link or is not the directory it is supposed to be.
            raise RefusedEntry(
                f"it, or a directory on the way down to it, is a symlink or is not a "
                f"directory ({exc}); a link where this task's own state belongs points "
                "at bytes it does not own"
            ) from exc
        raise RefusedEntry(f"it could not be opened ({exc})") from exc
    try:
        st = os.fstat(fd)
    except OSError as exc:  # pragma: no cover - fstat on a fresh descriptor
        os.close(fd)
        raise RefusedEntry(f"its shape could not be read ({exc})") from exc
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise RefusedEntry(
            f"it is not a regular file (mode {st.st_mode:#o}); a directory, socket or "
            "FIFO holds no transcript bytes to upload"
        )
    return Snapshot(
        fh=os.fdopen(fd, "rb"),
        fingerprint=Fingerprint(inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns),
    )


def _live_transcripts(settings: Settings) -> list[Path]:
    """The ``.jsonl`` files directly under the sessions directory.

    A missing directory yields nothing: a task that has served no turn has no sessions
    directory yet, and that is a first boot rather than a fault.
    """
    try:
        entries = sorted(os.scandir(settings.sessions_dir), key=lambda e: e.name)
    except FileNotFoundError:
        return []
    found: list[Path] = []
    for entry in entries:
        if not entry.name.endswith(keys.TRANSCRIPT_SUFFIX):
            continue
        # ``follow_symlinks=False`` so a link to a directory is not read as one file;
        # the shape is decided again on the descriptor, and this only decides what to
        # put in the list.
        if entry.is_dir(follow_symlinks=False):
            continue
        found.append(Path(entry.path))
    return found


def _archived_segments(settings: Settings) -> tuple[list[Path], list[tuple[str, str]]]:
    """Every file under the archive directory, at any depth, and any tree refusal.

    Walked rather than globbed at one level because rotation is free to nest, and a
    segment missed here is the older half of a conversation lost at the next task
    replacement.

    The chain down to the archive directory is opened first with every link refused. It
    has to be, because ``os.walk``'s ``followlinks=False`` governs directories it FINDS
    and not the root it is given: a link planted at ``sessions/archive`` is descended,
    and then every regular file behind it is a file with a key of its own and no reason
    to be in this bucket. When the chain is refused nothing under it is listed, and the
    refusal is returned so the cycle ends loudly instead of quietly backing up less.
    """
    root = settings.archive_dir
    try:
        os.close(_descend(settings.data_home, root.relative_to(settings.data_home).parts))
    except FileNotFoundError:
        return [], []
    except OSError as exc:
        return [], [
            (
                root.name,
                f"the archive directory, or a directory above it, could not be opened "
                f"as a real directory inside the data home ({exc}); nothing under it is "
                "listed, because a link there points at files this task does not own",
            )
        ]
    found: list[Path] = []
    for parent, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            found.append(Path(parent) / name)
    return found, []


@dataclass(frozen=True)
class BackupSet:
    """What one cycle should upload, in two phases, and what it already could not reach.

    The two lists are separate because the authority files are POINTERS: they name the
    transcripts, so they are only true once those transcripts are in the bucket. Holding
    them in their own phase is what lets the cycle publish them last, and withhold them
    entirely when a transcript did not make it.

    The refusals belong here rather than being discovered later because some of them are
    decided while LISTING, not while opening: a linked archive directory means a whole
    subtree is not enumerated, and that has to reach the cycle as a refusal. A set that
    returned only items would report a short cycle as a complete one.
    """

    data: list[tuple[str, Path]]
    authority: list[tuple[str, Snapshot]]
    authority_gone: list[str]
    refused: list[tuple[str, str]]

    def close_authority(self) -> None:
        """Release the authority descriptors, uploaded or not.

        The withheld path never uploads them, so closing cannot live at the upload site.
        """
        for _key, snapshot in self.authority:
            snapshot.close()


def objects_to_back_up(settings: Settings) -> BackupSet:
    """The cycle's two phases: every transcript, then the authority files that name them.

    The authority files are OPENED FIRST, before a single transcript is listed, and
    uploaded from those descriptors at the end of the cycle. Opening is what fixes the
    instant they describe: a descriptor's bounded length is the file as it was at open
    time, so the pair is one coherent snapshot of the index taken BEFORE the enumeration
    it indexes. Reading them at upload time instead let a slot table flushed during the
    cycle name a transcript that cycle never listed -- an index pointing at bytes that
    are not in the bucket.

    They are uploaded last for the same reason they are opened first. An index newer than
    the transcripts it names sends the front to an absent object, and the front reads that
    as a conversation that never had history: a live conversation served empty, with
    nothing raised. An index OLDER than the transcripts is the harmless direction, because
    every slot it names already has its bytes there and a transcript it does not name yet
    is unreferenced rather than misread.

    A missing authority file is not a failure. On a first boot the backend has not written
    one yet, and there is no index to preserve.
    """
    refused: list[tuple[str, str]] = []
    authority: list[tuple[str, Snapshot]] = []
    gone: list[str] = []
    for name in keys.AUTHORITY_NAMES:
        path = settings.config_dir / name
        try:
            snapshot = open_snapshot(path, root=settings.data_home)
        except RefusedEntry as exc:
            log.error("backup: refusing %s -- %s", name, exc)
            refused.append((name, str(exc)))
            continue
        if snapshot is None:
            log.info("backup: %s is not there yet; there is no index to preserve", name)
            gone.append(name)
            continue
        authority.append((keys.authority_key(settings, name), snapshot))
    archived, archive_refused = _archived_segments(settings)
    refused.extend(archive_refused)
    data = [
        (keys.data_key(settings, path), path) for path in _live_transcripts(settings) + archived
    ]
    return BackupSet(data=data, authority=authority, authority_gone=gone, refused=refused)


def run_cycle(
    settings: Settings,
    store: ObjectStore,
    *,
    state: dict[str, Fingerprint],
) -> CycleResult:
    """Upload everything in the set that has changed. Raise if anything was refused.

    *state* is read and written in place, so the caller keeps one map across cycles and
    an object unchanged since its last successful upload is not sent again.

    The authority phase runs only when the transcript phase reached everything it was
    asked for. A cycle that could not commit one transcript leaves the authority files
    as the last complete cycle wrote them, which is an older but coherent pair, rather
    than advancing the index past the bytes.

    A failure between the two authority PUTs leaves the bucket holding one file from this
    cycle's snapshot and one from an earlier cycle's. That skew is in the harmless
    direction -- both were opened before this cycle's enumeration, so neither names a
    transcript that is not in the bucket -- and it is not silent: the failure is a refusal,
    the cycle raises on it, and the next cycle publishes the pair together. What it costs
    is one interval in which the two files disagree about which slots exist.
    """
    result = CycleResult()
    plan = objects_to_back_up(settings)
    result.refused.extend(plan.refused)
    result.gone.extend(plan.authority_gone)
    try:
        _upload_phase(plan.data, settings=settings, store=store, state=state, result=result)
        if result.refused:
            for key, _snapshot in plan.authority:
                result.withheld.append(key)
            log.error(
                "backup: %d object(s) refused, so the authority files are NOT published "
                "this cycle -- the pair in the bucket stays at the last complete cycle "
                "rather than naming transcripts that are not there",
                len(result.refused),
            )
        else:
            _commit_authority(plan.authority, store=store, state=state, result=result)
    finally:
        plan.close_authority()
    log.info("backup: cycle complete -- %s", result.summary())
    if not result.complete:
        raise BackupIncomplete(result)
    return result


def _commit_authority(
    items: list[tuple[str, Snapshot]],
    *,
    store: ObjectStore,
    state: dict[str, Fingerprint],
    result: CycleResult,
) -> None:
    """Upload the already-open authority snapshots, recording each verdict in *result*.

    Separate from the transcript phase because these descriptors are opened by the plan,
    before the enumeration, and are closed by the caller whether or not this runs.
    """
    for key, snapshot in items:
        _commit_one(
            key, snapshot, name=key.rsplit("/", 1)[-1], store=store, state=state, result=result
        )


def _upload_phase(
    items: list[tuple[str, Path]],
    *,
    settings: Settings,
    store: ObjectStore,
    state: dict[str, Fingerprint],
    result: CycleResult,
) -> None:
    """Upload one phase of the set, recording every entry's verdict in *result*."""
    for key, path in items:
        try:
            snapshot = open_snapshot(path, root=settings.data_home)
        except RefusedEntry as exc:
            log.error("backup: refusing %s -- %s", path.name, exc)
            result.refused.append((path.name, str(exc)))
            continue
        if snapshot is None:
            log.info("backup: %s is gone; nothing to upload for it", path.name)
            result.gone.append(path.name)
            continue
        try:
            _commit_one(key, snapshot, name=path.name, store=store, state=state, result=result)
        finally:
            snapshot.close()


def _commit_one(
    key: str,
    snapshot: Snapshot,
    *,
    name: str,
    store: ObjectStore,
    state: dict[str, Fingerprint],
    result: CycleResult,
) -> None:
    """Send ONE open snapshot, or record why it was not sent. Never closes the descriptor.

    The caller owns the descriptor, because the two phases acquire it at different times:
    the transcript phase opens one per object as it goes, and the authority pair was opened
    by the plan before anything was enumerated.
    """
    if state.get(key) == snapshot.fingerprint:
        result.unchanged.append(key)
        return
    if snapshot.size > MAX_OBJECT_BYTES:
        # Uploaded anyway: skipping it is the data loss this design exists to
        # prevent. The warning is the point -- the front refuses to restore an
        # object this large, so the pair is honest but incomplete for this one
        # conversation, and an operator has to hear that from the writer rather
        # than from a customer's failed turn.
        log.warning(
            "backup: %s is %d B, above the %d B ceiling the restore side will "
            "read. It is uploaded, and a turn continuing this conversation on a "
            "replaced task will be refused rather than served an empty history.",
            name,
            snapshot.size,
            MAX_OBJECT_BYTES,
        )
        result.above_ceiling.append(key)
    try:
        store.put(key, snapshot.fh, snapshot.size)
    except StoreUnusable:
        # Not recorded as this object's refusal and not retried: the bucket
        # itself cannot be written, so every remaining object in this cycle and
        # every later cycle meets the same answer. It leaves here whole so the
        # process can end on it.
        raise
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        log.error("backup: PUT failed for %s -- %s", name, exc)
        result.refused.append((name, f"the upload failed ({exc})"))
        return
    state[key] = snapshot.fingerprint
    result.uploaded.append(key)
