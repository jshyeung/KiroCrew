"""Whether a spawning parent may keep working before it yields.

The spawn receipt tells the parent whether it may do a short, bounded step of
its own non-overlapping work before ending its turn, or must yield at once.
Only dashboard-owned turns have the busy-turn completion queue that makes the
first safe.
"""

from __future__ import annotations

from typing import Any


def parent_work_supported(state: Any, parent_session: str) -> bool:
    """Only dashboard-owned turns have the verified busy-turn completion queue.

    Channel-only, nested and background callers retain their yield boundary.
    A channel linked to a dashboard slot uses the same queue as dashboard chat.
    """
    if not parent_session or parent_session.startswith(("subagent:", "cron:", "hook:")):
        return False
    from kiro_crew.dashboard.chat_utils import effective_session_key

    slots = getattr(state, "_slots", None)
    return isinstance(slots, dict) and any(
        effective_session_key(slot) == parent_session for slot in slots.values()
    )
