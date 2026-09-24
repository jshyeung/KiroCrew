"""The subagent panel's durable rebuild source.

The panel's live source is gateway memory, so a replacement gateway process has
nothing to replay for the runs it never tracked. These tests pin the persisted
fallback that answers for them, and -- just as importantly -- pin what it must
refuse to invent: a slot-tracked native card, a run whose memory mode is not
persistent, a second copy of a run the live manager still holds, and a failure
for a run the tombstone records as a user stop.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from kiro_crew.dashboard.state import (
    NATIVE_SUBAGENT_TERMINAL_TTL_SECS,
    PERSISTED_SUBAGENT_REPLAY_KEEP,
    PERSISTED_SUBAGENT_REPLAY_MAX_AGE_SECS,
)
from kiro_crew.dashboard.ws import (
    build_persisted_subagent_frame,
    persisted_replay_app_matches,
)
from kiro_crew.subagent_persistence import (
    _PANEL_AGENT_CAP,
    _PANEL_APP_CAP,
    _PANEL_CANDIDATE_MULTIPLE,
    _PANEL_ERROR_CAP,
    _PANEL_RESULT_CAP,
    _PANEL_TASK_CAP,
    _PANEL_TRUNC_MARKER,
    classify_persisted_ending,
    read_panel_records,
)

DAY = 86_400.0
CORRUPT = "__corrupt__"
TRUNC = _PANEL_TRUNC_MARKER


@pytest.fixture()
def agent_root(tmp_path, monkeypatch):
    """Point persistence at a registry below this test's temp directory."""
    root = tmp_path / "subagents"
    root.mkdir()
    monkeypatch.setattr("kiro_crew.subagent_persistence._SUBAGENTS_DIR", root)
    return root


def write_record(
    root,
    agent_id: str,
    *,
    task: str = "summarise the changelog",
    agent: str = "kirocrew",
    parent_session: str = "dashboard:chat-1",
    started: float | None = None,
    memory_mode: str = "persistent",
    app: str = "",
    tombstone_cause: str | None = "delivered",
    outcome: str | None = None,
    detail: str | None = None,
    died: float | None = None,
    result: str | None = None,
    mtime: float | None = None,
) -> None:
    """Write one run folder the way a real run leaves it behind.

    ``tombstone_cause=None`` writes no tombstone; ``CORRUPT`` writes one that is
    present and unparseable. Those are different states and the reader tells them
    apart, so the helper has to be able to produce both.
    """
    moment = time.time()
    folder = root / agent_id
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        "id": agent_id,
        "task": task,
        "agent": agent,
        "parent_session": parent_session,
        "started": moment - 120 if started is None else started,
        "status": "running",
        "turns": 2,
        "memory_mode": memory_mode,
        "execution_context": {"memory_mode": memory_mode},
        "app": app,
        "updated_at": moment - 60,
    }
    (folder / "state.json").write_text(json.dumps(state), encoding="utf-8")
    if result is not None:
        (folder / "result.txt").write_text(result, encoding="utf-8")
    if tombstone_cause == CORRUPT:
        (folder / "tombstone.json").write_text("{not json", encoding="utf-8")
    elif tombstone_cause is not None:
        tombstone: dict = {
            "id": agent_id,
            "cause": tombstone_cause,
            "recovery_action": "delivered",
            "started": state["started"],
            "died": moment - 60 if died is None else died,
        }
        if outcome is not None:
            tombstone["outcome"] = outcome
        if detail is not None:
            tombstone["detail"] = detail
        (folder / "tombstone.json").write_text(json.dumps(tombstone), encoding="utf-8")
    if mtime is not None:
        os.utime(folder, (mtime, mtime))


def ids(records) -> list[str]:
    return [record["id"] for record in records]


class TestLiveStateWins:
    """The safety boundary: a disk record never displaces a tracked run.

    This is the property the whole fallback rests on. If disk could speak for an
    id the manager holds, a stale folder would overwrite the live card of a run
    still streaming -- turning a rebuild aid into a source of wrong answers.
    """

    def test_excluded_id_is_not_returned(self, agent_root):
        write_record(agent_root, "liveone")
        write_record(agent_root, "deadone")
        records = read_panel_records(
            keep=PERSISTED_SUBAGENT_REPLAY_KEEP,
            max_age_secs=DAY,
            exclude_ids={"liveone"},
        )
        assert ids(records) == ["deadone"]

    def test_every_id_excluded_yields_nothing(self, agent_root):
        write_record(agent_root, "aaa111")
        write_record(agent_root, "bbb222")
        records = read_panel_records(
            keep=PERSISTED_SUBAGENT_REPLAY_KEEP,
            max_age_secs=DAY,
            exclude_ids={"aaa111", "bbb222"},
        )
        assert records == []

    def test_exclusion_is_by_id_not_by_folder_order(self, agent_root):
        """A newer excluded folder must not shadow an older admissible one."""
        moment = time.time()
        write_record(agent_root, "newlive", mtime=moment - 10)
        write_record(agent_root, "olddead", mtime=moment - 1000)
        records = read_panel_records(
            keep=PERSISTED_SUBAGENT_REPLAY_KEEP,
            max_age_secs=DAY,
            exclude_ids={"newlive"},
        )
        assert ids(records) == ["olddead"]

    def test_an_excluded_folder_is_never_even_read(self, agent_root, monkeypatch):
        """Exclusion happens during the walk, so a live run costs no state read."""
        from kiro_crew import subagent_persistence

        write_record(agent_root, "liveone")
        reads: list[str] = []
        real = subagent_persistence.read_state
        monkeypatch.setattr(
            subagent_persistence,
            "read_state",
            lambda aid: (reads.append(aid), real(aid))[1],
        )
        assert read_panel_records(keep=10, max_age_secs=DAY, exclude_ids={"liveone"}) == []
        assert reads == []


class TestNativeCardsAreNotInvented:
    """Native slot-tracked cards have no durable record anywhere.

    ``create_agent_folder`` is reached only from the manager's admission pump, so
    a native run writes no folder. The fallback discovers records by walking
    folders, and these pin that the walk cannot conjure one: a native card that
    vanished with its gateway stays gone rather than reappearing as a card the
    panel cannot address.
    """

    def test_empty_registry_yields_nothing(self, agent_root):
        records = read_panel_records(keep=PERSISTED_SUBAGENT_REPLAY_KEEP, max_age_secs=DAY)
        assert records == []

    def test_manager_record_is_found_while_native_id_is_absent(self, agent_root):
        """A folder-backed run appears; a native id with no folder never does."""
        write_record(agent_root, "manager1")
        records = read_panel_records(keep=PERSISTED_SUBAGENT_REPLAY_KEEP, max_age_secs=DAY)
        assert ids(records) == ["manager1"]
        assert not any(record["id"].startswith("native:") for record in records)

    def test_a_folder_holding_only_a_result_is_not_a_record(self, agent_root):
        """No ``state.json`` means no identity, so there is nothing to replay."""
        folder = agent_root / "resultonly"
        folder.mkdir()
        (folder / "result.txt").write_text("orphan output", encoding="utf-8")
        records = read_panel_records(keep=PERSISTED_SUBAGENT_REPLAY_KEEP, max_age_secs=DAY)
        assert records == []


class TestNonPersistentRunsAreNotInvented:
    """A non-persistent run keeps its state in memory and writes no folder.

    The mode check is belt and braces on top of that: a folder whose record
    spells ``incognito`` or ``temporary`` is skipped explicitly, so the exclusion
    holds however the folder came to exist. Failing closed costs one card;
    failing open would put a private run's task text on screen.
    """

    @pytest.mark.parametrize("mode", ["incognito", "temporary", "ephemeral", ""])
    def test_non_persistent_mode_is_skipped(self, agent_root, mode):
        write_record(agent_root, "private1", memory_mode=mode)
        records = read_panel_records(keep=PERSISTED_SUBAGENT_REPLAY_KEEP, max_age_secs=DAY)
        assert records == []

    def test_mode_missing_from_the_record_is_skipped(self, agent_root):
        """An unstated mode is not assumed persistent."""
        folder = agent_root / "nomode"
        folder.mkdir()
        (folder / "state.json").write_text(
            json.dumps(
                {
                    "id": "nomode",
                    "task": "t",
                    "parent_session": "dashboard:chat-1",
                    "started": time.time() - 60,
                }
            ),
            encoding="utf-8",
        )
        records = read_panel_records(keep=PERSISTED_SUBAGENT_REPLAY_KEEP, max_age_secs=DAY)
        assert records == []

    def test_nested_execution_mode_alone_is_honoured(self, agent_root):
        """The mode is read from the execution record when the top level omits it."""
        folder = agent_root / "nested"
        folder.mkdir()
        (folder / "state.json").write_text(
            json.dumps(
                {
                    "id": "nested",
                    "task": "t",
                    "parent_session": "dashboard:chat-1",
                    "started": time.time() - 60,
                    "execution_context": {"memory_mode": "incognito"},
                }
            ),
            encoding="utf-8",
        )
        records = read_panel_records(keep=PERSISTED_SUBAGENT_REPLAY_KEEP, max_age_secs=DAY)
        assert records == []


class TestEndingClassification:
    """One classifier answers for both the list and the single-card read.

    The tombstone carries the run's own ``outcome``, so reading only ``cause``
    reports a routine user stop as a failure -- and the run's specific reason
    lives in ``detail`` while ``cause`` is a coarse bucket.
    """

    def test_a_recorded_user_stop_stays_a_stop(self, agent_root):
        write_record(agent_root, "stop111", tombstone_cause="user_stop", outcome="stopped")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["outcome"] == "stopped"
        assert record["stopped"] is True
        assert record["error"] == ""

    def test_a_recorded_outcome_outranks_the_cause_bucket(self, agent_root):
        """``cause`` says reaped; the outcome the run recorded says stopped."""
        write_record(agent_root, "rank111", tombstone_cause="reaped", outcome="stopped")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["outcome"] == "stopped"

    def test_a_recorded_failure_prefers_its_detail_over_the_bucket(self, agent_root):
        write_record(
            agent_root,
            "det111",
            tombstone_cause="error",
            outcome="failed",
            detail="provider returned 503",
        )
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["outcome"] == "failed"
        assert record["error"] == "provider returned 503"

    def test_a_failure_without_detail_names_its_cause(self, agent_root):
        write_record(agent_root, "orph111", tombstone_cause="gateway_restart")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["outcome"] == "failed"
        assert record["error"] == "Orphaned: gateway_restart"

    def test_delivered_reads_as_completed(self, agent_root):
        write_record(agent_root, "done111", tombstone_cause="delivered")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["outcome"] == "completed"
        assert record["error"] == ""
        assert record["stopped"] is False

    def test_an_unreadable_tombstone_is_an_unknown_cause(self, agent_root):
        """An ending WAS recorded and cannot be read -- not the same as none."""
        write_record(agent_root, "corr111", tombstone_cause=CORRUPT)
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["outcome"] == "failed"
        assert record["error"] == "Orphaned (unknown cause)"

    def test_no_tombstone_is_reported_without_an_error(self, agent_root):
        """Nothing recorded an ending; the reconciler writes one at startup."""
        write_record(agent_root, "none111", tombstone_cause=None)
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["outcome"] == "completed"
        assert record["error"] == ""

    def test_an_unrecognised_outcome_falls_back_to_the_cause(self, agent_root):
        write_record(agent_root, "junk111", tombstone_cause="gateway_restart", outcome="banana")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["outcome"] == "failed"
        assert record["error"] == "Orphaned: gateway_restart"

    def test_classifier_is_callable_on_its_own_for_the_single_card_read(self, agent_root):
        """The single-card endpoint shares this exact function, not a copy."""
        write_record(agent_root, "shar111", tombstone_cause="user_stop", outcome="stopped")
        assert classify_persisted_ending(agent_root / "shar111") == ("stopped", "", True)

    def test_classifier_takes_the_folder_so_one_response_resolves_it_once(self, agent_root):
        """Taking a path, not an id, is what lets a caller share its resolution."""
        write_record(agent_root, "path111", tombstone_cause="error", detail="boom")
        assert classify_persisted_ending(agent_root / "path111") == ("failed", "boom", False)
        assert classify_persisted_ending(agent_root / "absent") == ("completed", "", False)

    def test_elapsed_spans_start_to_death(self, agent_root):
        moment = time.time()
        write_record(agent_root, "span111", started=moment - 300, died=moment - 60)
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["elapsed"] == pytest.approx(240.0, abs=1.0)

    def test_death_before_start_yields_no_negative_elapsed(self, agent_root):
        moment = time.time()
        write_record(agent_root, "skew111", started=moment - 60, died=moment - 300)
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["elapsed"] == 0.0

    def test_task_and_agent_travel_with_the_record(self, agent_root):
        write_record(agent_root, "idy111", task="audit the gate", agent="kirocrew-worker")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["task"] == "audit the gate"
        assert record["agent"] == "kirocrew-worker"


class TestOwnership:
    def test_record_without_a_parent_is_dropped(self, agent_root):
        """An empty slot routes nowhere, and older clients read it as the active tab."""
        write_record(agent_root, "noown1", parent_session="")
        records = read_panel_records(keep=10, max_age_secs=DAY)
        assert records == []


class TestRetainedFieldsAreBounded:
    """A row count is not a memory bound: 50 rows of an unbounded task is unbounded.

    Every retained string carries its own named cap, and a value that was cut ends
    in the marker -- which travels inside the value to every reader, so no separate
    flag has to be carried and kept in step with it.
    """

    def test_task_is_clamped_and_says_it_was_cut(self, agent_root):
        write_record(agent_root, "big111", task="t" * (_PANEL_TASK_CAP + 500))
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert len(record["task"]) < _PANEL_TASK_CAP + 100
        assert record["task"].startswith("t" * 100)
        assert record["task"].endswith(TRUNC)

    def test_agent_is_clamped(self, agent_root):
        write_record(agent_root, "big222", agent="a" * (_PANEL_AGENT_CAP + 500))
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert len(record["agent"]) < _PANEL_AGENT_CAP + 100
        assert record["agent"].endswith(TRUNC)

    def test_error_detail_is_clamped(self, agent_root):
        write_record(
            agent_root,
            "big333",
            tombstone_cause="error",
            outcome="failed",
            detail="e" * (_PANEL_ERROR_CAP + 500),
        )
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert len(record["error"]) < _PANEL_ERROR_CAP + 100
        assert record["error"].endswith(TRUNC)

    def test_result_is_clamped_and_says_it_was_cut(self, agent_root):
        write_record(agent_root, "big444", result="x" * (_PANEL_RESULT_CAP + 5000))
        (record,) = read_panel_records(keep=10, max_age_secs=DAY, include_result=True)
        assert len(record["result"]) < _PANEL_RESULT_CAP + 100
        assert record["result"].endswith(TRUNC)

    def test_a_result_exactly_at_the_cap_is_not_marked_cut(self, agent_root):
        write_record(agent_root, "exact1", result="x" * _PANEL_RESULT_CAP)
        (record,) = read_panel_records(keep=10, max_age_secs=DAY, include_result=True)
        assert record["result"] == "x" * _PANEL_RESULT_CAP
        assert not record["result"].endswith(TRUNC)

    def test_ordinary_fields_carry_no_marker(self, agent_root):
        write_record(agent_root, "small1", task="short", result="also short")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY, include_result=True)
        assert record["task"] == "short"
        assert record["result"] == "also short"

    def test_the_scan_working_set_is_bounded_by_keep_not_folder_count(
        self, agent_root, monkeypatch
    ):
        """Unreadable candidates must not let the walk grow with the registry."""
        from kiro_crew import subagent_persistence

        moment = time.time()
        for index in range(60):
            write_record(
                agent_root,
                f"skip{index:04d}",
                memory_mode="incognito",
                mtime=moment - index,
            )
        reads: list[str] = []
        real = subagent_persistence.read_state
        monkeypatch.setattr(
            subagent_persistence,
            "read_state",
            lambda aid: (reads.append(aid), real(aid))[1],
        )
        records = read_panel_records(keep=2, max_age_secs=DAY)
        assert records == []
        assert len(reads) <= 2 * _PANEL_CANDIDATE_MULTIPLE


class TestBounds:
    def test_keep_caps_the_burst(self, agent_root):
        moment = time.time()
        for index in range(8):
            write_record(agent_root, f"agent{index:03d}", mtime=moment - index)
        records = read_panel_records(keep=3, max_age_secs=DAY)
        assert len(records) == 3

    def test_keep_of_zero_reads_nothing(self, agent_root):
        write_record(agent_root, "any111")
        assert read_panel_records(keep=0, max_age_secs=DAY) == []

    def test_newest_folders_are_preferred_when_the_cap_bites(self, agent_root):
        moment = time.time()
        write_record(agent_root, "newest", mtime=moment - 5)
        write_record(agent_root, "middle", mtime=moment - 500)
        write_record(agent_root, "oldest", mtime=moment - 5000)
        records = read_panel_records(keep=2, max_age_secs=DAY)
        assert ids(records) == ["newest", "middle"]

    def test_records_past_the_age_bound_are_dropped(self, agent_root):
        stale = time.time() - (3 * DAY)
        write_record(agent_root, "stale1", started=stale, died=stale, mtime=stale)
        records = read_panel_records(keep=10, max_age_secs=DAY)
        assert records == []

    def test_a_recently_touched_folder_is_still_judged_on_its_recorded_times(self, agent_root):
        """The folder's mtime orders the walk; the run's own times decide the window.

        A late write inside a folder -- a result chunk, a tombstone -- moves its
        mtime without moving the run. Folder mtime therefore cannot be the age
        decision, and this pins the check that is.
        """
        stale = time.time() - (3 * DAY)
        write_record(agent_root, "touched", started=stale, died=stale, mtime=time.time() - 5)
        records = read_panel_records(keep=10, max_age_secs=DAY)
        assert records == []

    def test_folders_outside_the_window_are_never_opened(self, agent_root, monkeypatch):
        """The age filter runs during the walk, so a stale folder costs no read."""
        from kiro_crew import subagent_persistence

        moment = time.time()
        write_record(agent_root, "fresh01", started=moment - 30, died=moment - 10, mtime=moment - 1)
        stale = moment - (5 * DAY)
        for index in range(30):
            write_record(
                agent_root,
                f"old{index:04d}",
                started=stale,
                died=stale,
                mtime=stale - index,
            )
        reads: list[str] = []
        real = subagent_persistence.read_state
        monkeypatch.setattr(
            subagent_persistence,
            "read_state",
            lambda aid: (reads.append(aid), real(aid))[1],
        )
        records = read_panel_records(keep=10, max_age_secs=DAY)
        assert ids(records) == ["fresh01"]
        assert reads == ["fresh01"]

    def test_a_record_inside_the_day_survives_the_native_hour(self, agent_root):
        """The whole reason this bound is its own number rather than the native TTL.

        A run that ended two hours before the gateway was replaced is outside the
        native terminal TTL and still inside this one. Sharing the hour-long
        constant would leave the panel empty in exactly the case the fallback
        exists to serve.
        """
        two_hours_ago = time.time() - 7200
        assert 7200 > NATIVE_SUBAGENT_TERMINAL_TTL_SECS
        write_record(
            agent_root, "twohr1", started=two_hours_ago, died=two_hours_ago, mtime=two_hours_ago
        )
        records = read_panel_records(
            keep=PERSISTED_SUBAGENT_REPLAY_KEEP,
            max_age_secs=PERSISTED_SUBAGENT_REPLAY_MAX_AGE_SECS,
        )
        assert ids(records) == ["twohr1"]

    def test_configured_bounds_are_the_approved_policy(self):
        assert PERSISTED_SUBAGENT_REPLAY_KEEP == 50
        assert PERSISTED_SUBAGENT_REPLAY_MAX_AGE_SECS == DAY


class TestResultText:
    def test_result_is_withheld_unless_asked_for(self, agent_root):
        write_record(agent_root, "res111", result="the answer")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert "result" not in record

    def test_result_is_read_when_asked_for(self, agent_root):
        write_record(agent_root, "res222", result="the answer")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY, include_result=True)
        assert record["result"] == "the answer"

    def test_missing_result_file_is_empty_not_an_error(self, agent_root):
        write_record(agent_root, "res333")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY, include_result=True)
        assert record["result"] == ""

    def test_a_sensitive_result_path_is_not_read(self, agent_root, monkeypatch):
        """The same guard the single-agent read applies, at the one place a file opens."""
        write_record(agent_root, "res555", result="secret material")
        monkeypatch.setattr("kiro_crew.subagent_persistence.is_sensitive_path", lambda path: True)
        (record,) = read_panel_records(keep=10, max_age_secs=DAY, include_result=True)
        assert record["result"] == ""


class TestCorruptFolders:
    def test_unreadable_state_is_skipped_without_losing_the_rest(self, agent_root):
        moment = time.time()
        broken = agent_root / "broken"
        broken.mkdir()
        (broken / "state.json").write_text("{not json", encoding="utf-8")
        os.utime(broken, (moment - 1, moment - 1))
        write_record(agent_root, "intact", mtime=moment - 2)
        records = read_panel_records(keep=10, max_age_secs=DAY)
        assert ids(records) == ["intact"]

    def test_a_file_in_the_registry_is_not_a_record(self, agent_root):
        (agent_root / "stray.json").write_text("{}", encoding="utf-8")
        write_record(agent_root, "intact")
        records = read_panel_records(keep=10, max_age_secs=DAY)
        assert ids(records) == ["intact"]

    def test_absent_registry_reads_as_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "kiro_crew.subagent_persistence._SUBAGENTS_DIR", tmp_path / "never-created"
        )
        assert read_panel_records(keep=10, max_age_secs=DAY) == []


class TestReusedSlotKeyOwnership:
    """A slot key an app's run was recorded under can later belong to another app.

    Slot keys are caller-supplied and are not namespaced by app, and the per-frame
    scope gate authorizes against the slot's CURRENT owner -- which for a reused
    key is not the owner the run belonged to. Within the replay window that hands
    the new owner the previous owner's task, agent name and error text, delivered
    once with no recall. So the run's own recorded app is compared here.
    """

    def slot(self, app: str):
        return SimpleNamespace(_app=app)

    def state(self, slots: dict):
        return SimpleNamespace(_slots=slots)

    @pytest.mark.parametrize(
        "record_app, slot_app, allowed",
        [
            ("", "", True),
            ("appA", "appA", True),
            ("appA", "appB", False),
            ("appA", "", False),
            ("", "appB", False),
        ],
    )
    def test_the_recorded_app_must_equal_the_slots_current_owner(
        self, record_app, slot_app, allowed
    ):
        state = self.state({"chat-1": self.slot(slot_app)})
        record = {"id": "a", "app": record_app, "parent_session": "dashboard:chat-1"}
        assert persisted_replay_app_matches(state, "chat-1", record) is allowed

    def test_a_missing_slot_denies_because_there_is_no_owner_to_compare(self):
        state = self.state({})
        assert persisted_replay_app_matches(state, "chat-1", {"app": ""}) is False

    def test_an_empty_slot_key_denies(self):
        state = self.state({"chat-1": self.slot("")})
        assert persisted_replay_app_matches(state, "", {"app": ""}) is False

    def test_an_empty_slot_key_denies_even_if_a_slot_is_registered_under_it(self):
        """The empty key is refused on its own, not merely by finding no slot.

        A record whose parent resolves to nothing carries an empty slot, and a
        lookup would otherwise hand it whatever sits under that key.
        """
        state = self.state({"": self.slot("")})
        assert persisted_replay_app_matches(state, "", {"app": ""}) is False

    def test_a_missing_app_key_on_the_record_reads_as_no_app(self):
        """An older record with no app field is a run no app owns."""
        state = self.state({"chat-1": self.slot("")})
        assert persisted_replay_app_matches(state, "chat-1", {"id": "a"}) is True
        state_app = self.state({"chat-1": self.slot("appB")})
        assert persisted_replay_app_matches(state_app, "chat-1", {"id": "a"}) is False

    def test_a_slot_without_the_attribute_reads_as_no_app(self):
        state = self.state({"chat-1": SimpleNamespace()})
        assert persisted_replay_app_matches(state, "chat-1", {"app": ""}) is True
        assert persisted_replay_app_matches(state, "chat-1", {"app": "appA"}) is False


class TestRecordedApp:
    def test_the_records_app_comes_from_the_run_state(self, agent_root):
        write_record(agent_root, "app111", app="my-app")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["app"] == "my-app"

    def test_a_run_no_app_owns_carries_an_empty_app(self, agent_root):
        write_record(agent_root, "app222")
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["app"] == ""

    def test_the_nested_execution_app_is_used_when_the_top_level_omits_it(self, agent_root):
        folder = agent_root / "app333"
        folder.mkdir()
        (folder / "state.json").write_text(
            json.dumps(
                {
                    "id": "app333",
                    "task": "t",
                    "parent_session": "dashboard:chat-1",
                    "started": time.time() - 60,
                    "memory_mode": "persistent",
                    "execution_context": {"memory_mode": "persistent", "app": "nested-app"},
                }
            ),
            encoding="utf-8",
        )
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["app"] == "nested-app"

    def test_an_implausible_app_id_drops_the_record_rather_than_clamping_it(self, agent_root):
        """A clamped id would compare unequal while looking like a value."""
        write_record(agent_root, "app444", app="a" * (_PANEL_APP_CAP + 1))
        assert read_panel_records(keep=10, max_age_secs=DAY) == []

    def test_an_app_id_exactly_at_the_cap_is_kept_whole(self, agent_root):
        write_record(agent_root, "app555", app="a" * _PANEL_APP_CAP)
        (record,) = read_panel_records(keep=10, max_age_secs=DAY)
        assert record["app"] == "a" * _PANEL_APP_CAP


class TestReplayFrame:
    """The frame shape the panel's reducers consume."""

    def plain(self, text: str) -> str:
        return text

    def test_frame_is_a_terminal_subagent_event(self):
        frame = build_persisted_subagent_frame(
            {
                "id": "abc123",
                "task": "audit the gate",
                "agent": "kirocrew-worker",
                "parent_session": "dashboard:chat-1",
                "started": 100.0,
                "elapsed": 42.0,
                "outcome": "completed",
                "error": "",
                "stopped": False,
            },
            redact=self.plain,
        )
        assert frame["type"] == "subagent_done"
        data = frame["data"]
        assert data["id"] == "abc123"
        assert data["slot"] == "chat-1"
        assert data["elapsed"] == 42.0
        assert data["outcome"] == "completed"
        assert data["task"] == "audit the gate"
        assert data["agent"] == "kirocrew-worker"
        assert data["stopped"] is False

    def test_a_recorded_stop_reaches_the_frame(self):
        frame = build_persisted_subagent_frame(
            {
                "id": "a",
                "parent_session": "dashboard:chat-1",
                "outcome": "stopped",
                "error": "",
                "stopped": True,
            },
            redact=self.plain,
        )
        assert frame["data"]["stopped"] is True
        assert frame["data"]["outcome"] == "stopped"

    def test_no_error_is_sent_as_null_not_an_empty_string(self):
        frame = build_persisted_subagent_frame(
            {"id": "a", "parent_session": "dashboard:chat-1", "outcome": "completed", "error": ""},
            redact=self.plain,
        )
        assert frame["data"]["error"] is None

    def test_an_error_travels_through_the_callers_redactor(self):
        frame = build_persisted_subagent_frame(
            {
                "id": "a",
                "parent_session": "dashboard:chat-1",
                "outcome": "failed",
                "error": "Orphaned: gateway_restart",
            },
            redact=lambda text: text.upper(),
        )
        assert frame["data"]["error"] == "ORPHANED: GATEWAY_RESTART"

    def test_task_and_agent_travel_through_the_callers_redactor(self):
        frame = build_persisted_subagent_frame(
            {
                "id": "a",
                "parent_session": "dashboard:chat-1",
                "outcome": "completed",
                "error": "",
                "task": "secret",
                "agent": "worker",
            },
            redact=lambda text: f"[{text}]",
        )
        assert frame["data"]["task"] == "[secret]"
        assert frame["data"]["agent"] == "[worker]"

    def test_a_record_with_no_parent_yields_an_ownerless_frame(self):
        """Which the replay's own owner filter then drops, as it does for live frames."""
        frame = build_persisted_subagent_frame(
            {"id": "a", "parent_session": "", "outcome": "completed", "error": ""},
            redact=self.plain,
        )
        assert frame["data"]["slot"] == ""
