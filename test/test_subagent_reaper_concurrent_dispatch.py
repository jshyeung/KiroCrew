"""Regression: the reaper dispatches its due force-reaps CONCURRENTLY.

Each ``_force_reap`` runs ``reset()``, which awaits ``provider.shutdown()`` while
holding the session-manager global lock — the same lock every turn-start
(``get_or_create``) contends. When the reaper reaped its due agents inline and
serially, back-to-back stuck teardowns starved turn-start on ALL sessions for up
to ``_RESET_TIMEOUT`` *per* stuck child, which surfaced as a multi-minute UI
"hang" during an autopilot fan-out (kirodotdev/KiroCrew#14715).

The fix collects the sweep's due reaps and dispatches them with
``asyncio.gather``. This test pins that contract: given two over-deadline
agents, BOTH reaps must be in flight at once. Against the old serial code the
second reap cannot begin until the first returns, so the ``started`` event below
never fires and the ``wait_for`` times out — i.e. the test fails on the un-fixed
code, which is what earns it its place.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.subagent import SubagentInfo, SubagentManager


def _make_manager() -> SubagentManager:
    mgr = SubagentManager(
        sessions=MagicMock(), ctx_builder=MagicMock(), max_concurrent=8
    )
    # Silence every non-reap sweep the loop body performs so the test drives
    # only the agent scan + reap dispatch.
    mgr._conv_registry_rebuilt = True
    mgr._sample_live_costs = MagicMock()
    mgr._refresh_learned_cost = MagicMock()
    mgr._sweep_stuck_waves_async = AsyncMock()
    mgr._sweep_digest_holds_async = AsyncMock()
    mgr._sweep_conversations = MagicMock()
    mgr._taskq_pump = MagicMock()
    mgr._admission = MagicMock()
    mgr._maybe_flag_stall = AsyncMock()
    mgr._is_startup_stalled = MagicMock(return_value=False)
    return mgr


def _due_info(agent_id: str) -> SubagentInfo:
    info = SubagentInfo(id=agent_id, task="t", agent="")
    # Started long ago so ``elapsed > _default_timeout`` — a wall-clock reap.
    info.started = 0.0
    info.done = False
    return info


@pytest.mark.asyncio
async def test_reaper_dispatches_due_reaps_concurrently():
    mgr = _make_manager()
    mgr._agents["a1111111"] = _due_info("a1111111")
    mgr._agents["b2222222"] = _due_info("b2222222")

    started = asyncio.Event()  # set once BOTH reaps have begun
    inflight = 0
    max_inflight = 0
    release = asyncio.Event()  # no reap may finish until both have started

    async def fake_force_reap(agent_id, info, elapsed, reason=None):
        nonlocal inflight, max_inflight
        inflight += 1
        max_inflight = max(max_inflight, inflight)
        if inflight >= 2:
            started.set()
        # Serial code blocks here on the first reap and never lets the second
        # start, so ``started`` never fires and the wait_for below times out.
        await release.wait()
        inflight -= 1

    mgr._force_reap = fake_force_reap

    # The reaper sleeps ``_REAPER_INTERVAL`` (default 60s) at the top of each
    # pass. ``bind_component_globals`` injects that name into the reaper impl's
    # own ``__globals__`` (not the module dict), so zero it there for an instant
    # first pass and restore it after.
    import kiro_crew.subagent_manager.monitoring as monitoring_mod

    reaper_globals = monitoring_mod.OrphanStallMonitor._reaper_loop_impl.__globals__
    prev_interval = reaper_globals.get("_REAPER_INTERVAL")
    reaper_globals["_REAPER_INTERVAL"] = 0

    loop_task = asyncio.create_task(mgr._reaper_loop())
    try:
        # Both reaps must be in flight together; if the dispatch were serial
        # this wait would time out because the second reap never starts.
        await asyncio.wait_for(started.wait(), timeout=5.0)
        assert max_inflight == 2, "reaps were not dispatched concurrently"
    finally:
        release.set()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
        reaper_globals["_REAPER_INTERVAL"] = prev_interval
