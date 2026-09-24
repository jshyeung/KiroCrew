"""Pins for sharing one ``AcpRuntime`` process across top-level chat sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kiro_crew.acp.chat_runtime_pool import (
    ChatRuntimeKey,
    ChatRuntimePool,
    eligible_for_chat_sharing,
)


class FakeRuntime:
    """Stands in for ``AcpRuntime``: the pool only probes ``is_alive``/``pid``."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def die(self) -> None:
        self._alive = False


def spawner(*runtimes: FakeRuntime):
    """An ``acquire`` spawn callback handing back *runtimes* in order."""
    pending = list(runtimes)
    calls: list[FakeRuntime] = []

    async def spawn() -> FakeRuntime:
        rt = pending.pop(0)
        calls.append(rt)
        return rt

    spawn.calls = calls  # type: ignore[attr-defined]
    return spawn


def a_key(**overrides) -> ChatRuntimeKey:
    base = dict(
        work_dir="/home/u/.kirocrew/workspace",
        agent="kirocrew",
        sandbox_mode="auto",
        extra_env={"A": "1", "B": "2"},
        acp_backend="kiro",
        tool_search=None,
        member_context=False,
        memory_mode="persistent",
        shared_scratch=None,
        mcp_gateway_overlay=None,
        mcp_gateway_socket=None,
    )
    base.update(overrides)
    return ChatRuntimeKey.build(**base)


class TestChatSharingEligibility:
    """D10: only a dashboard chat slot in persistent memory mode may share."""

    def test_dashboard_chat_slot_is_eligible(self):
        assert eligible_for_chat_sharing(
            session_key="dashboard:chat-12-1790000000",
            memory_mode="persistent",
            member_context=False,
            sharing_enabled=True,
        )

    def test_bare_chat_slot_key_is_eligible(self):
        assert eligible_for_chat_sharing(
            session_key="chat-12-1790000000",
            memory_mode="persistent",
            member_context=False,
            sharing_enabled=True,
        )

    @pytest.mark.parametrize("mode", ["incognito", "temporary"])
    def test_non_persistent_never_shares(self, mode):
        assert not eligible_for_chat_sharing(
            session_key="dashboard:chat-12-1790000000",
            memory_mode=mode,
            member_context=False,
            sharing_enabled=True,
        )

    def test_member_session_never_shares(self):
        assert not eligible_for_chat_sharing(
            session_key="dashboard:chat-12-1790000000",
            memory_mode="persistent",
            member_context=True,
            sharing_enabled=True,
        )

    def test_flag_off_disables_sharing(self):
        assert not eligible_for_chat_sharing(
            session_key="dashboard:chat-12-1790000000",
            memory_mode="persistent",
            member_context=False,
            sharing_enabled=False,
        )

    @pytest.mark.parametrize(
        "key",
        [
            "cron:nightly-digest",
            "hook:pre-commit",
            "task:runner-7",
            "subagent:abc123",
            "telegram:kirocrew:direct:8743158320",
            None,
            "",
        ],
    )
    def test_non_chat_origins_keep_their_own_runtime(self, key):
        assert not eligible_for_chat_sharing(
            session_key=key,
            memory_mode="persistent",
            member_context=False,
            sharing_enabled=True,
        )


class TestChatRuntimeKey:
    """D4: the key is every process-level spawn input, and nothing per-session."""

    def test_identical_inputs_compare_equal_and_hash_equal(self):
        assert a_key() == a_key()
        assert len({a_key(), a_key()}) == 1

    def test_env_order_does_not_fragment_the_pool(self):
        assert a_key(extra_env={"A": "1", "B": "2"}) == a_key(extra_env={"B": "2", "A": "1"})

    @pytest.mark.parametrize(
        "field,value",
        [
            ("work_dir", "/home/u/oss/other-worktree"),
            ("agent", "kirocrew-lite"),
            ("sandbox_mode", "strict"),
            ("extra_env", {"A": "9"}),
            ("acp_backend", "kas"),
            ("member_context", True),
            ("memory_mode", "incognito"),
            ("shared_scratch", Path("/scratch/tree-a")),
            ("mcp_gateway_overlay", "/overlay/a.json"),
            ("mcp_gateway_socket", "/run/gw.sock"),
            ("model", "some-model-id"),
        ],
    )
    def test_every_process_level_field_splits_the_key(self, field, value):
        assert a_key() != a_key(**{field: value})

    def test_path_and_string_spellings_of_work_dir_agree(self):
        assert a_key(work_dir=Path("/home/u/.kirocrew/workspace")) == a_key()

    def test_a_trailing_separator_is_the_same_directory(self):
        assert a_key(work_dir="/home/u/.kirocrew/workspace/") == a_key()

    @pytest.mark.parametrize(
        "spelling",
        ["/home/u/.kirocrew/workspace", "/home/u/.kirocrew/workspace/"],
    )
    def test_freeze_path_normalizes_every_spelling_of_one_directory(self, spelling):
        from kiro_crew.acp.chat_runtime_pool import _freeze_path

        assert _freeze_path(spelling) == _freeze_path(Path(spelling))
        assert _freeze_path(spelling) == _freeze_path("/home/u/.kirocrew/workspace")

    @pytest.mark.parametrize("unset", [None, ""])
    def test_freeze_path_reports_an_unset_field_as_empty(self, unset):
        from kiro_crew.acp.chat_runtime_pool import _freeze_path

        assert _freeze_path(unset) == ""

    def test_different_directories_still_split_the_key(self):
        from kiro_crew.acp.chat_runtime_pool import _freeze_path

        assert _freeze_path("/home/u/a") != _freeze_path("/home/u/b")

    def test_tool_search_settings_are_part_of_the_key(self):
        class TS:
            def __init__(self, enabled, min_pct, min_tokens):
                self.enabled = enabled
                self.min_pct = min_pct
                self.min_tokens = min_tokens

        on = a_key(tool_search=TS(True, 5, 50000))
        off = a_key(tool_search=TS(False, 5, 50000))
        assert on != off
        assert on != a_key(tool_search=None)


class TestRefcountedRegistry:
    """D6: a runtime lives while any session holds it, and D2 caps the sharing."""

    def test_second_compatible_session_joins_one_process(self):
        pool = ChatRuntimePool()
        rt = FakeRuntime()
        spawn = spawner(rt)

        first = asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        second = asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))

        assert first.joined is False
        assert second.joined is True
        assert first.runtime is second.runtime
        assert len(spawn.calls) == 1
        assert second.sessions_on_runtime == 2
        assert pool.sessions_on(rt) == ["chat-1", "chat-2"]

    def test_incompatible_session_gets_its_own_process(self):
        pool = ChatRuntimePool()
        spawn = spawner(FakeRuntime(1), FakeRuntime(2))

        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        other = asyncio.run(pool.acquire(a_key(work_dir="/home/u/oss/wt"), "chat-2", spawn, cap=10))

        assert other.joined is False
        assert len(spawn.calls) == 2

    def test_cap_overflow_spawns_another_runtime(self):
        pool = ChatRuntimePool()
        first_rt, second_rt = FakeRuntime(1), FakeRuntime(2)
        spawn = spawner(first_rt, second_rt)

        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=2))
        asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=2))
        third = asyncio.run(pool.acquire(a_key(), "chat-3", spawn, cap=2))

        assert third.runtime is second_rt
        assert third.joined is False
        assert pool.sessions_on(first_rt) == ["chat-1", "chat-2"]
        assert pool.sessions_on(second_rt) == ["chat-3"]

    def test_reacquire_by_the_same_session_does_not_double_count(self):
        pool = ChatRuntimePool()
        rt = FakeRuntime()
        spawn = spawner(rt)

        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        again = asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))

        assert again.runtime is rt
        assert pool.sessions_on(rt) == ["chat-1"]
        assert len(spawn.calls) == 1

    def test_only_the_last_release_hands_back_the_runtime_to_kill(self):
        pool = ChatRuntimePool()
        rt = FakeRuntime()
        spawn = spawner(rt)
        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))

        assert asyncio.run(pool.release("chat-1")) is None
        assert pool.runtime_for("chat-2") is rt
        assert asyncio.run(pool.release("chat-2")) is rt
        assert pool.runtime_for("chat-2") is None

    def test_releasing_an_unheld_session_is_a_no_op(self):
        pool = ChatRuntimePool()
        assert asyncio.run(pool.release("chat-never-held")) is None

    def test_shared_pids_maps_one_process_to_its_sessions(self):
        pool = ChatRuntimePool()
        spawn = spawner(FakeRuntime(777))
        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))

        assert pool.shared_pids() == {777: ["chat-1", "chat-2"]}


class TestRootPreference:
    """D3: a session started from a root prefers the root's process."""

    def test_root_runtime_is_preferred_over_an_older_compatible_one(self):
        pool = ChatRuntimePool()
        older, root_rt = FakeRuntime(1), FakeRuntime(2)
        spawn = spawner(older, root_rt)

        asyncio.run(pool.acquire(a_key(), "chat-older", spawn, cap=10))
        asyncio.run(pool.acquire(a_key(work_dir="/home/u/oss/wt"), "chat-root", spawn, cap=10))

        child = asyncio.run(
            pool.acquire(
                a_key(work_dir="/home/u/oss/wt"),
                "chat-child",
                spawn,
                cap=10,
                prefer_session_key="chat-root",
            )
        )
        assert child.runtime is root_rt

    def test_a_full_root_does_not_block_placement_elsewhere(self):
        pool = ChatRuntimePool()
        root_rt, spare = FakeRuntime(1), FakeRuntime(2)
        spawn = spawner(root_rt, spare)

        asyncio.run(pool.acquire(a_key(), "chat-root", spawn, cap=1))
        child = asyncio.run(
            pool.acquire(a_key(), "chat-child", spawn, cap=1, prefer_session_key="chat-root")
        )

        assert child.runtime is spare
        assert child.joined is False

    def test_an_incompatible_root_is_not_used(self):
        pool = ChatRuntimePool()
        root_rt, fresh = FakeRuntime(1), FakeRuntime(2)
        spawn = spawner(root_rt, fresh)

        asyncio.run(pool.acquire(a_key(agent="kirocrew-lite"), "chat-root", spawn, cap=10))
        child = asyncio.run(
            pool.acquire(a_key(), "chat-child", spawn, cap=10, prefer_session_key="chat-root")
        )

        assert child.runtime is fresh


class TestRuntimeDeathRecovery:
    """D7: no session stays bound to a dead runtime."""

    def test_forget_unbinds_every_session_on_the_lost_runtime(self):
        pool = ChatRuntimePool()
        rt = FakeRuntime()
        spawn = spawner(rt)
        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))
        asyncio.run(pool.acquire(a_key(), "chat-3", spawn, cap=10))

        rt.die()
        orphaned = asyncio.run(pool.forget(rt))

        assert orphaned == ["chat-1", "chat-2", "chat-3"]
        for key in orphaned:
            assert pool.runtime_for(key) is None

    def test_orphaned_sessions_land_together_on_one_replacement(self):
        pool = ChatRuntimePool()
        dead, replacement = FakeRuntime(1), FakeRuntime(2)
        spawn = spawner(dead, replacement)
        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))

        dead.die()
        asyncio.run(pool.forget(dead))

        first_back = asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        second_back = asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))

        assert first_back.runtime is replacement
        assert first_back.joined is False
        assert second_back.runtime is replacement
        assert second_back.joined is True
        assert len(spawn.calls) == 2

    def test_a_dead_runtime_is_never_handed_to_a_new_session(self):
        pool = ChatRuntimePool()
        dead, fresh = FakeRuntime(1), FakeRuntime(2)
        spawn = spawner(dead, fresh)
        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))

        dead.die()
        joined = asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))

        assert joined.runtime is fresh
        assert joined.joined is False
        assert pool.runtime_for("chat-1") is None

    def test_a_dead_runtime_is_dropped_from_the_pid_map(self):
        pool = ChatRuntimePool()
        rt = FakeRuntime(999)
        spawn = spawner(rt, FakeRuntime(1000))
        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        rt.die()

        asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))

        assert 999 not in pool.shared_pids()


class KillableRuntime(FakeRuntime):
    """A runtime that records whether anything killed it."""

    def __init__(self, pid: int = 4242) -> None:
        super().__init__(pid)
        self.kills: list[str] = []

    async def kill(self, *, expected: bool = False, reason: str = "") -> None:
        self.kills.append(reason)
        self._alive = False


class FakeHandle:
    """The only handle method the pooled shutdown branch reaches."""

    def __init__(self) -> None:
        self.destroyed = 0
        self.is_turn_active = False

    async def destroy(self) -> None:
        self.destroyed += 1


def pooled_provider(pool: ChatRuntimePool, runtime, session_key: str):
    from kiro_crew.acp.session_provider import AcpSessionProvider

    handle = FakeHandle()
    provider = AcpSessionProvider(
        handle,  # type: ignore[arg-type]
        runtime,
        session_key=session_key,
        runtime_release=lambda: pool.release(session_key),
    )
    return provider, handle


class TestPooledShutdownRefcount:
    """D6 at the teardown seam: the pool, not the provider, decides the kill."""

    def test_a_joining_session_leaving_does_not_kill_the_shared_process(self):
        pool = ChatRuntimePool()
        rt = KillableRuntime()
        spawn = spawner(rt)
        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))

        provider, handle = pooled_provider(pool, rt, "chat-2")
        asyncio.run(provider.shutdown())

        assert handle.destroyed == 1
        assert rt.kills == []
        assert rt.is_alive()
        assert pool.runtime_for("chat-1") is rt

    def test_the_founder_leaving_first_does_not_kill_it_either(self):
        pool = ChatRuntimePool()
        rt = KillableRuntime()
        spawn = spawner(rt)
        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))
        asyncio.run(pool.acquire(a_key(), "chat-2", spawn, cap=10))

        provider, _ = pooled_provider(pool, rt, "chat-1")
        asyncio.run(provider.shutdown())

        assert rt.kills == []
        assert rt.is_alive()

    def test_the_last_holder_leaving_kills_the_process(self):
        pool = ChatRuntimePool()
        rt = KillableRuntime()
        spawn = spawner(rt)
        asyncio.run(pool.acquire(a_key(), "chat-1", spawn, cap=10))

        provider, handle = pooled_provider(pool, rt, "chat-1")
        asyncio.run(provider.shutdown())

        assert handle.destroyed == 1
        assert len(rt.kills) == 1
        assert not rt.is_alive()

    def test_a_failing_release_still_destroys_this_session_handle(self):
        from kiro_crew.acp.session_provider import AcpSessionProvider

        rt = KillableRuntime()
        handle = FakeHandle()

        async def boom():
            raise RuntimeError("registry unavailable")

        provider = AcpSessionProvider(
            handle,  # type: ignore[arg-type]
            rt,
            session_key="chat-1",
            runtime_release=boom,
        )
        asyncio.run(provider.shutdown())

        assert handle.destroyed == 1
        assert rt.kills == []
