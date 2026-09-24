"""Putting bytes at a path that must not already hold them.

Both directions of the durability pair write a file this way, and both write it for
the same reason: the bytes came from the bucket, something else may be writing the
same name, and an existing file is always the copy to keep. So the mechanism lives
here once.

Only the MECHANISM is shared. Each caller keeps its own refusal message and its own
exception type, because what an existing file or a redirected parent MEANS differs:
for the front it decides whether a customer's turn may be served, and for the restore
step it decides whether the task may boot. A shared helper that also decided how to
complain would have to be told, which is the same thing as leaving it to the caller.

The three properties, all load-bearing:

* **Atomic.** The bytes land in a temporary file in the same directory, are flushed and
  fsynced, and then appear at the target under one name. A crash cannot leave a
  truncated file for something else to read or append to.
* **Never an overwrite.** ``os.link`` refuses an existing target, so "do not clobber"
  is a filesystem guarantee rather than a check with a window after it. It also refuses
  a symlink at the target without following it, so a link planted there cannot receive
  the bytes.
* **No temporary left behind.** The temporary is unlinked on every path out, including
  the one where the link succeeded and it has served its purpose.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["link_new"]


def link_new(path: Path, data: bytes, *, prefix: str) -> bool:
    """Create *path* holding *data*. ``True`` if it landed, ``False`` if it existed.

    ``False`` is not an error and is deliberately not raised: an existing file is the
    newer copy in both callers, so the answer they need is "was it already there", not
    a failure to handle. Any other problem -- an unwritable directory, a full disk --
    raises ``OSError``, for the caller to translate into its own refusal.

    *prefix* names the temporary, so a caller's own leftover is recognisable as its
    own. It is required rather than defaulted, because the two callers write into
    different directories and a shared temp name is a shared thing to clean up.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, str(path))
        except FileExistsError:
            return False
        return True
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
