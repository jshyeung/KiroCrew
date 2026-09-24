"""The apps tree is masked from every cron child that runs model-supplied code.

``<config_dir>/apps/<name>/.app_secret`` is a bearer credential, not a marker:
``dashboard.token_auth.validate_app_secret`` compares it and issues that app's scoped
token. Both cron exec paths run model-supplied code -- a script body under the
LLM-writeable ``crons/`` dir, and a ``command`` string the cron tool accepts verbatim --
so neither child may reach any app's secret.

What these tests pin is the WIRING, the mask target's existence, and the data window each
installed app keeps inside the mask. The two sandbox builders' own handling of a
caller-supplied ``extra_hidden_dirs`` tree and of a window inside it is pinned by
``test_sandbox_private_window_in_caller_mask.py``; asserting it again here would test
the builders twice and the call site not at all.

Every assertion is over the arguments the call site hands ``wrap_argv`` and over the
directory it leaves on disk. Nothing here spawns a child: the recording stub raises
after capturing, which the runners convert into a failed job.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from kiro_crew import sandbox
from kiro_crew.cron_script import run_command_sandboxed, run_script_sandboxed
from kiro_crew.sandbox import (
    SandboxCeilingUnsealable,
    SandboxUnavailableError,
    materialize_caller_masked_dir,
)

_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits / bind-mount mask")


@pytest.fixture(autouse=True)
def _cron_caller_is_named(named_cron_caller):
    """Cron spawn bookkeeping wants a nameable caller; these tests assume it."""


@pytest.fixture(autouse=True)
def cron_home(monkeypatch, tmp_path):
    """Point ``cron_script.config_dir`` at the per-test patched home.

    Same redirect as ``test_cron_secret_env.py``: scripts live under
    ``<home>/.kirocrew/crons`` so the resolver and the mask target agree on one root,
    and a test that does not patch ``Path.home`` falls back to a per-test tmp dir rather
    than the operator's real home, which this fixture must never create or key.
    """
    real_home = Path.home()
    fallback = tmp_path / "kirocrew-home-fallback"
    fallback.mkdir(parents=True, exist_ok=True)

    def _dir() -> Path:
        home = Path.home()
        root = fallback if home == real_home else home / ".kirocrew"
        # The real ``config_dir`` resolves AND creates the home on every call, so a stub
        # that did not would invent an absent-parent case production cannot reach.
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr("kiro_crew.cron_script.config_dir", _dir)
    return _dir


def _make_script(tmp_path: Path, body: str = "def run(ctx): pass\n") -> Path:
    crons_dir = tmp_path / ".kirocrew" / "crons"
    crons_dir.mkdir(parents=True, exist_ok=True)
    script = crons_dir / "job.py"
    script.write_text(body, encoding="utf-8")
    return script


class _Recorder:
    """Captures one ``wrap_argv`` call, then refuses so no child is spawned.

    ``SandboxUnavailableError`` is the refusal both runners already convert into a
    failed-job dict, so the capture costs no subprocess and no special-case handling.
    """

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.calls = 0

    def __call__(self, argv, **kwargs):
        self.calls += 1
        self.kwargs = dict(kwargs)
        raise SandboxUnavailableError(
            "recorded by the test stub", "test-stub", "the call site's arguments"
        )

    @property
    def hidden(self) -> list[str]:
        return list(self.kwargs.get("extra_hidden_dirs") or ())

    @property
    def windows(self) -> list[str]:
        return list(self.kwargs.get("extra_private_dirs") or ())


def _record(monkeypatch, *, mask_applies: bool = True) -> _Recorder:
    """Stub ``wrap_argv`` with a recorder, and state whether the mask is CARRIED.

    The two are one setting, not two. A recorder stands in for a wrap that happens, and
    the call sites materialize their mask target only when ``credential_mask_applies``
    says the spawn will carry the mask -- so a recorder left beside a false predicate
    would describe a host that wraps the argv and drops the mask, which is not a host
    the sandbox has. The default states the ordinary host; ``mask_applies=False`` names
    the one where the child comes back unwrapped.
    """
    recorder = _Recorder()
    monkeypatch.setattr("kiro_crew.cron_script.wrap_argv", recorder)
    monkeypatch.setattr("kiro_crew.cron_script.credential_mask_applies", lambda mode: mask_applies)
    return recorder


class TestTheScriptCronChildCannotReachAnyAppSecret:
    def test_an_ungranted_run_masks_the_apps_tree(self, tmp_path, monkeypatch):
        """The ordinary case, and the one that hid nothing at all.

        An ungranted script body is the least trustworthy cron source -- the ``crons/``
        dir is LLM-writeable by design -- and it was the branch passing an EMPTY hidden
        list, so every installed app's secret was readable.
        """
        recorder = _record(monkeypatch)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert recorder.calls == 1
        assert str(tmp_path / ".kirocrew" / "apps") in recorder.hidden
        assert result["status"] == "error"

    def test_a_granted_run_masks_the_apps_tree_beside_what_it_already_hid(
        self, tmp_path, monkeypatch
    ):
        """A grant's isolation is the STRICTER of the two, so it cannot be the branch
        that keeps the credentials. Its existing entries stay: the live ``crons/`` dir
        and the script's own parent close the mutable-sibling import route, which the
        app mask has nothing to do with.
        """
        recorder = _record(monkeypatch)
        script = _make_script(tmp_path)
        # A grant that fails its precheck never reaches wrap_argv, so drive the granted
        # branch through the pin the runner itself computes.
        from kiro_crew.cron_script import compute_secret_env_pin
        from kiro_crew.secrets import SecretVault

        SecretVault(tmp_path / ".kirocrew").set_sync("slack-sandbox", "xoxb-script")
        grant = {"MY_SANDBOX_TOKEN": "slack-sandbox"}
        spec = f"{script}:run"
        with patch("pathlib.Path.home", return_value=tmp_path):
            pin = compute_secret_env_pin(spec, "", job_id="job1", grant=grant)
            run_script_sandboxed(spec, "job1", timeout=30, secret_env=grant, secret_env_pin=pin)

        assert recorder.calls == 1
        hidden = recorder.hidden
        assert str(tmp_path / ".kirocrew" / "apps") in hidden
        assert str(tmp_path / ".kirocrew" / "crons") in hidden
        assert str(Path(str(script)).resolve().parent) in hidden

    def test_the_containing_tree_is_masked_not_a_list_of_installed_apps(
        self, tmp_path, monkeypatch
    ):
        """An app installed WHILE this run executes must be covered too.

        A mask is applied to the paths named at spawn and is never recomputed for a live
        child, so a list of leaves read out of ``apps/`` cannot name an app installed a
        second later -- that app's secret would stay readable for the rest of the
        child's life. Naming the directory is what covers whatever appears under it, so
        this asserts the absence of per-app entries as well as the tree's presence.
        """
        apps = tmp_path / ".kirocrew" / "apps"
        for name in ("dev-fleet", "meetings"):
            (apps / name).mkdir(parents=True)
            (apps / name / ".app_secret").write_text("s3cret", encoding="utf-8")
        recorder = _record(monkeypatch)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        hidden = recorder.hidden
        assert str(apps) in hidden
        below = [h for h in hidden if h.startswith(str(apps) + os.sep)]
        assert below == [], f"per-app entries cannot cover a later install: {below}"


class TestTheCommandCronChildCannotReachAnyAppSecret:
    def test_a_command_run_masks_the_apps_tree(self, tmp_path, monkeypatch):
        """The sibling exec path, and the reason it is in scope.

        ``run_command_sandboxed``'s own comment records that the command string is fully
        model-supplied, so the two cron exec paths share one trust level and a control
        on only one of them is bypassable by choosing the other.
        """
        recorder = _record(monkeypatch)
        # The brace-expansion shell probe wraps an argv of its own; pin the shell so the
        # recorded call is the command's, not the probe's.
        monkeypatch.setattr("kiro_crew.cron_script._resolve_command_shell", lambda: "/bin/sh")
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = run_command_sandboxed("echo hi", timeout=30, job_id="job1")

        assert recorder.calls == 1
        assert str(tmp_path / ".kirocrew" / "apps") in recorder.hidden
        assert result["status"] == "error"


class TestTheMaskTargetExistsBeforeTheMaskIsBuilt:
    @_POSIX_ONLY
    def test_an_absent_apps_tree_is_created_so_the_bind_has_a_name(self, tmp_path, monkeypatch):
        """The sharpest case: a home where no app has ever been installed.

        The Linux launcher's ``SENSITIVE_DIRS`` loop is guarded on ``isdir``, so an
        absent target gets no empty bind over it and the FIRST-EVER install then appears
        inside this running child's view -- with its secret in it.
        """
        apps = tmp_path / ".kirocrew" / "apps"
        recorder = _record(monkeypatch)
        script = _make_script(tmp_path)
        assert not apps.exists()
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert apps.is_dir()
        assert stat.S_IMODE(apps.stat().st_mode) == 0o700
        assert str(apps) in recorder.hidden

    @_POSIX_ONLY
    def test_an_existing_apps_tree_keeps_its_mode_and_its_contents(self, tmp_path, monkeypatch):
        """The installed apps live here and the gateway owns the directory.

        Tightening its mode, or touching what is in it, would change an unrelated
        component's behaviour for a reason having nothing to do with this spawn.
        """
        apps = tmp_path / ".kirocrew" / "apps"
        (apps / "dev-fleet").mkdir(parents=True)
        (apps / "dev-fleet" / ".app_secret").write_text("keep-me", encoding="utf-8")
        apps.chmod(0o755)
        _record(monkeypatch)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert stat.S_IMODE(apps.stat().st_mode) == 0o755
        assert (apps / "dev-fleet" / ".app_secret").read_text(encoding="utf-8") == "keep-me"

    @_POSIX_ONLY
    def test_a_symlinked_apps_leaf_refuses_the_run_before_any_child_starts(
        self, tmp_path, monkeypatch
    ):
        """``isdir`` follows a link, so the mask would cover the link's TARGET while the
        replaceable name stayed writable. Refuse instead of masking the wrong inode --
        and refuse before ``wrap_argv``, so nothing is spawned unmasked.
        """
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        apps = tmp_path / ".kirocrew" / "apps"
        apps.parent.mkdir(parents=True, exist_ok=True)
        apps.symlink_to(elsewhere, target_is_directory=True)
        recorder = _record(monkeypatch)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert result["status"] == "error"
        assert recorder.calls == 0, "a refused mask must not reach the spawn"


class TestAHostCarryingNoMaskGainsNoNewRefusal:
    """The boundary the other direction: where the mask is DROPPED, nothing is created.

    ``credential_mask_applies`` is false exactly where ``wrap_argv`` hands back an
    unwrapped child, so every ``extra_hidden_dirs`` entry is dropped there. Creating or
    validating a mask target on such a host confines nothing, and its only possible
    effect is to fail a spawn the sandbox would otherwise allow -- so the materializer
    is not reached at all, and whatever the sandbox layer itself decides is what the
    operator hears.
    """

    @_POSIX_ONLY
    def test_the_apps_tree_is_not_created_when_the_mask_is_dropped(self, tmp_path, monkeypatch):
        apps = tmp_path / ".kirocrew" / "apps"
        recorder = _record(monkeypatch, mask_applies=False)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert recorder.calls == 1, "the spawn still reaches the sandbox layer"
        assert not apps.exists(), "an unwrapped child gains nothing from an empty dir"

    @_POSIX_ONLY
    def test_the_sandbox_layers_own_refusal_is_what_reaches_the_caller(self, tmp_path, monkeypatch):
        """The remedy an operator acts on lives in the sandbox layer's refusal.

        A create error standing in front of it would replace an actionable message with
        an incidental one on the very hosts that need the actionable one.
        """
        recorder = _record(monkeypatch, mask_applies=False)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert recorder.calls == 1
        assert result["status"] == "error"
        assert "recorded by the test stub" in result["error"]

    @_POSIX_ONLY
    def test_a_symlinked_apps_leaf_does_not_refuse_a_run_that_carries_no_mask(
        self, tmp_path, monkeypatch
    ):
        """The squat is only dangerous to a mask, and no mask is built here.

        Refusing on it would fail a spawn for a hazard this host cannot suffer, which is
        the new refusal the masking change deliberately does not add.
        """
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        apps = tmp_path / ".kirocrew" / "apps"
        apps.parent.mkdir(parents=True, exist_ok=True)
        apps.symlink_to(elsewhere, target_is_directory=True)
        recorder = _record(monkeypatch, mask_applies=False)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert recorder.calls == 1, "no mask to protect, so no refusal to add"
        assert apps.is_symlink(), "the squatted name is left exactly as found"


class TestEachInstalledAppKeepsItsDataInsideTheMask:
    """Masking the tree also covers ``apps/<app>/data``, which is not a credential.

    ``apps.manager.app_data_dir`` documents that directory as an app's persistence root,
    and the Linux mask binds a WRITABLE empty directory over the tree -- so a child
    writing there is told the write succeeded and the bytes go away with the namespace.
    Success with no data and no error is the worst shape a failure can take, so each
    installed app's data directory is passed as a private window: the mask and every
    ``.app_secret`` stay denied, and only those directories stay live on their real
    inodes.

    The builders' own handling of a window inside a caller mask is pinned by
    ``test_sandbox_private_window_in_caller_mask.py``; what these tests pin is that this
    call site composes the two arguments the builders already honour.
    """

    @staticmethod
    def _install(tmp_path: Path, *names: str) -> Path:
        apps = tmp_path / ".kirocrew" / "apps"
        for name in names:
            (apps / name).mkdir(parents=True, exist_ok=True)
            (apps / name / ".app_secret").write_text("secret", encoding="utf-8")
        return apps

    @_POSIX_ONLY
    def test_an_ungranted_script_run_keeps_every_installed_apps_data_live(
        self, tmp_path, monkeypatch
    ):
        apps = self._install(tmp_path, "alpha", "beta")
        recorder = _record(monkeypatch)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert str(apps) in recorder.hidden
        assert recorder.windows == [str(apps / "alpha" / "data"), str(apps / "beta" / "data")]

    @_POSIX_ONLY
    def test_a_command_run_keeps_them_live_too(self, tmp_path, monkeypatch):
        """The command path is the same trust level, so it gets the same view."""
        apps = self._install(tmp_path, "alpha")
        recorder = _record(monkeypatch)
        # Pin the shell for the reason the sibling command test gives: the
        # brace-expansion probe wraps an argv of its own.
        monkeypatch.setattr("kiro_crew.cron_script._resolve_command_shell", lambda: "/bin/sh")
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_command_sandboxed("echo hi", timeout=30, job_id="job1")

        assert str(apps) in recorder.hidden
        assert recorder.windows == [str(apps / "alpha" / "data")]

    @_POSIX_ONLY
    def test_no_window_names_a_credential(self, tmp_path, monkeypatch):
        """The window is the DATA directory, so no spelling of it reaches a secret.

        A ``.app_secret`` sits beside ``data`` rather than inside it, so the ancestor
        mask still denies every one of them. This is the property that makes the window
        safe to open at all, which is why it is asserted rather than assumed.
        """
        self._install(tmp_path, "alpha", "beta")
        recorder = _record(monkeypatch)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert recorder.windows, "the case is vacuous if no window was opened"
        for window in recorder.windows:
            assert os.path.basename(window) == "data"
            assert ".app_secret" not in window

    @_POSIX_ONLY
    def test_an_absent_data_dir_is_created_so_a_first_ever_write_survives(
        self, tmp_path, monkeypatch
    ):
        """The launcher's window loop is guarded on ``isdir`` like the mask loop.

        An app that has never written data has no ``data`` directory, so the window
        would be skipped in silence and the mask would cover the path -- losing exactly
        the first write, which is the one an app is most likely to make from a cron.
        """
        apps = self._install(tmp_path, "alpha")
        assert not (apps / "alpha" / "data").exists()
        recorder = _record(monkeypatch)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert (apps / "alpha" / "data").is_dir()
        assert recorder.windows == [str(apps / "alpha" / "data")]

    @_POSIX_ONLY
    def test_a_dropped_mask_opens_no_window_and_creates_nothing(self, tmp_path, monkeypatch):
        """No mask, nothing to keep live inside it, and no directory to invent."""
        apps = self._install(tmp_path, "alpha")
        recorder = _record(monkeypatch, mask_applies=False)
        script = _make_script(tmp_path)
        with patch("pathlib.Path.home", return_value=tmp_path):
            run_script_sandboxed(f"{script}:run", "job1", timeout=30)

        assert recorder.calls == 1
        assert recorder.windows == []
        assert not (apps / "alpha" / "data").exists()


class TestTheDataWindowEnumeration:
    """``app_data_window_targets`` degrades per app, and never fails the spawn.

    One app's on-disk layout must not stop every cron on the host, and the fail-closed
    answer for an unverifiable path is to withhold its view rather than to grant one.
    """

    @_POSIX_ONLY
    def test_an_absent_tree_yields_nothing(self, tmp_path):
        assert sandbox.app_data_window_targets(str(tmp_path / "nope")) == ()

    @_POSIX_ONLY
    def test_a_plain_file_where_the_tree_should_be_yields_nothing(self, tmp_path):
        target = tmp_path / "apps"
        target.write_text("not a tree", encoding="utf-8")
        assert sandbox.app_data_window_targets(str(target)) == ()

    @_POSIX_ONLY
    def test_a_symlinked_app_directory_is_skipped_rather_than_traversed(self, tmp_path):
        """Following it would re-expose the link's target inside the masked tree."""
        apps = tmp_path / "apps"
        (apps / "real").mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (apps / "linked").symlink_to(elsewhere, target_is_directory=True)

        assert sandbox.app_data_window_targets(str(apps)) == (str(apps / "real" / "data"),)
        assert not (elsewhere / "data").exists(), "nothing is created through the link"

    @_POSIX_ONLY
    def test_one_unusable_app_withholds_only_its_own_window(self, tmp_path):
        """A file squatting one ``data`` name leaves the other apps' windows intact."""
        apps = tmp_path / "apps"
        for name in ("alpha", "beta", "gamma"):
            (apps / name).mkdir(parents=True)
        (apps / "beta" / "data").write_text("squatter", encoding="utf-8")

        assert sandbox.app_data_window_targets(str(apps)) == (
            str(apps / "alpha" / "data"),
            str(apps / "gamma" / "data"),
        )

    @_POSIX_ONLY
    def test_a_dangling_link_at_a_data_name_withholds_that_window(self, tmp_path):
        apps = tmp_path / "apps"
        (apps / "alpha").mkdir(parents=True)
        (apps / "alpha" / "data").symlink_to(tmp_path / "missing", target_is_directory=True)

        assert sandbox.app_data_window_targets(str(apps)) == ()

    @_POSIX_ONLY
    def test_an_existing_data_dir_keeps_its_contents(self, tmp_path):
        apps = tmp_path / "apps"
        data = apps / "alpha" / "data"
        data.mkdir(parents=True)
        (data / "state.json").write_text("{}", encoding="utf-8")

        assert sandbox.app_data_window_targets(str(apps)) == (str(data),)
        assert (data / "state.json").read_text(encoding="utf-8") == "{}"


class TestALifecycleMoveAsideIsNotAnApp:
    """The apps tree holds the manager's own move-asides beside the installed apps.

    ``kiro_crew.apps.manager`` parks an app's data and secret under dot-prefixed names in
    the SAME directory while an install, update or uninstall is in flight, and leaves them
    there when the operation crashes. Because a window is CREATED when absent, an
    enumeration that counted one of those as an app would write a ``data`` child inside
    it, and the restore leg moves the move-aside onto the app's real persistence root.
    """

    #: Every dot-prefixed shape ``apps/manager.py`` parks beside an app, for ``alpha``.
    _MOVE_ASIDES = (
        ".alpha-data-tmp",
        ".alpha-secret-tmp",
        ".alpha-update-old-4321-beef",
        ".alpha-deps-doomed2",
    )

    @_POSIX_ONLY
    def test_a_data_move_aside_gets_no_window_and_gains_no_data_child(self, tmp_path):
        apps = tmp_path / "apps"
        parked = apps / ".alpha-data-tmp"
        parked.mkdir(parents=True)

        assert sandbox.app_data_window_targets(str(apps)) == ()
        assert not (parked / "data").exists()

    @_POSIX_ONLY
    @pytest.mark.parametrize("parked_name", _MOVE_ASIDES)
    def test_every_move_aside_shape_is_skipped(self, tmp_path, parked_name):
        apps = tmp_path / "apps"
        (apps / parked_name).mkdir(parents=True)

        assert sandbox.app_data_window_targets(str(apps)) == ()
        assert not (apps / parked_name / "data").exists()

    @_POSIX_ONLY
    def test_a_real_app_beside_a_move_aside_still_gets_its_window(self, tmp_path):
        """The skip is the name contract, not a reason to withhold a live app's window."""
        apps = tmp_path / "apps"
        (apps / "alpha").mkdir(parents=True)
        (apps / ".beta-data-tmp").mkdir()

        assert sandbox.app_data_window_targets(str(apps)) == (str(apps / "alpha" / "data"),)

    @_POSIX_ONLY
    def test_an_update_restores_preserved_data_without_a_nested_copy(self, tmp_path):
        """The harm, as the manager's own move-aside and restore legs perform it."""
        apps = tmp_path / "apps"
        data = apps / "alpha" / "data"
        data.mkdir(parents=True)
        (data / "keep.txt").write_text("user rows", encoding="utf-8")

        # update_app parks the data tree, replaces the app directory, then moves the
        # parked tree back onto the fresh copy. A cron spawn lands inside that window.
        parked = apps / ".alpha-data-tmp"
        data.rename(parked)
        sandbox.app_data_window_targets(str(apps))
        parked.rename(data)

        assert (data / "keep.txt").read_text(encoding="utf-8") == "user rows"
        assert not (data / "data").exists()
        assert sorted(p.name for p in data.iterdir()) == ["keep.txt"]


@_POSIX_ONLY
class TestOnlyTheAppNameContractEarnsAWindow:
    """A window is built for a name ``app_name_error`` admits, and for no other.

    The tier-collision test compares path TEXT, so it matches one spelling of a hidden
    app. On a case-insensitive filesystem -- the macOS default -- ``apps/AWS-Control``
    resolves to the ``apps/aws-control`` the tier hides, and a window carrying that
    directory would re-bind an owner-authorization store read-write for a cron child
    running model-supplied code. ``app_name_error`` forces lowercase kebab-case, so the
    spelling no admission path ever issued never reaches the comparison.

    POSIX-only for a reason beyond convention: one case creates a directory named
    ``nul``, which Windows reserves as a device name and refuses.
    """

    #: Lowercase kebab names the contract still refuses, each for its own reason.
    _REFUSED_KEBAB = ("system", "library", "registry", "nul")

    def test_a_case_variant_of_a_tier_hidden_app_gets_no_window(self, tmp_path):
        apps = tmp_path / "apps"
        squatter = apps / "AWS-Control"
        squatter.mkdir(parents=True)

        assert sandbox.app_data_window_targets(str(apps)) == ()
        assert not (squatter / "data").exists()

    def test_an_uppercase_name_gets_no_window(self, tmp_path):
        apps = tmp_path / "apps"
        (apps / "Alpha").mkdir(parents=True)

        assert sandbox.app_data_window_targets(str(apps)) == ()
        assert not (apps / "Alpha" / "data").exists()

    @pytest.mark.parametrize("refused", _REFUSED_KEBAB)
    def test_a_reserved_or_unportable_name_gets_no_window(self, tmp_path, refused):
        """The gate is the whole contract, not just its character pattern."""
        apps = tmp_path / "apps"
        (apps / refused).mkdir(parents=True)

        assert sandbox.app_data_window_targets(str(apps)) == ()
        assert not (apps / refused / "data").exists()

    def test_a_name_the_contract_admits_still_gets_its_window(self, tmp_path):
        """The refusals are a filter on names, not a reason to withhold a real app's data."""
        apps = tmp_path / "apps"
        (apps / "alpha").mkdir(parents=True)
        (apps / "Alpha-Two").mkdir()

        assert sandbox.app_data_window_targets(str(apps)) == (str(apps / "alpha" / "data"),)


class TestATierMaskedAppDataTreeGetsNoWindow:
    """An app whose data the tier masks in its own right keeps it masked.

    ``apps/aws-control/data`` and ``apps/meetings/data/edits`` are hidden in every mode
    -- one holds live AWS credentials, the other owner-authorization bits -- and they sit
    INSIDE the tree this call site masks. A window is re-bound read-write over that
    tree, and it is staged from the real inode before any mask is applied, so a window
    at or above one of those leaves would carry it back out: the ordinary
    ``.app_secret`` denial would be traded for a credential store.

    Two relationships have to be refused, and asking whether the window merely sits
    under SOME hidden parent catches neither: the window EQUALS the hidden leaf
    (``aws-control``), or the window CONTAINS it (``meetings``, whose ``data`` holds the
    hidden ``data/edits``).
    """

    @staticmethod
    def _install(tmp_path: Path, *names: str) -> Path:
        apps = tmp_path / ".kirocrew" / "apps"
        for name in names:
            (apps / name / "data").mkdir(parents=True, exist_ok=True)
        return apps

    @_POSIX_ONLY
    def test_a_window_equal_to_a_hidden_leaf_is_withheld(self, tmp_path):
        apps = self._install(tmp_path, "alpha", "aws-control")
        with patch("pathlib.Path.home", return_value=tmp_path):
            windows = sandbox.app_data_window_targets(str(apps))

        assert windows == (str(apps / "alpha" / "data"),)

    @_POSIX_ONLY
    def test_a_window_containing_a_hidden_leaf_is_withheld(self, tmp_path):
        apps = self._install(tmp_path, "alpha", "meetings")
        with patch("pathlib.Path.home", return_value=tmp_path):
            windows = sandbox.app_data_window_targets(str(apps))

        assert windows == (str(apps / "alpha" / "data"),)

    @_POSIX_ONLY
    def test_the_launcher_keeps_the_leaf_denied_and_opens_no_window_for_it(self):
        """The gate refuses the collision even when a caller names it directly.

        This is the guarantee that does not depend on the enumeration above: any caller
        handing the launcher such a window gets it dropped, and the leaf stays in the
        masked list, while an ordinary app's window on the same tree survives.
        """
        home = os.path.expanduser("~")
        apps = os.path.join(home, ".kirocrew", "apps")
        leaf = os.path.join(apps, "aws-control", "data")
        ordinary = os.path.join(apps, "alpha", "data")
        script = sandbox._build_launcher_script(
            "cc", extra_hidden_dirs=(apps,), extra_private_dirs=(leaf, ordinary)
        )
        masked = json.loads(re.search(r"SENSITIVE_DIRS = (\[.*?\])\n", script, re.S).group(1))
        windows = json.loads(re.search(r"PRIVATE_DIRS = (\[.*?\])\n", script, re.S).group(1))

        assert leaf not in windows
        assert leaf in masked
        assert ordinary in windows


class TestAnAppNameCannotForgeTheOperatorsLog:
    """An app DIRECTORY NAME is agent-created, so it never reaches the log raw.

    ``gateway.log`` uses a plain formatter, so a name holding a newline or an ANSI escape
    could forge log lines and rewrite the operator's terminal on every cron run. Two
    things keep it out. The name contract admits lowercase kebab-case only, so such a
    name earns no window and reaches no refusal branch -- that is what these cases
    assert, including that the contract wins over the later per-app refusal the same
    directory also triggers. And every log site that does interpolate an entry's name
    wraps it in ``safe_terminal_line``, asserted over the source below because no name
    the contract admits can exercise it.
    """

    @_POSIX_ONLY
    def test_a_newline_in_an_app_name_does_not_reach_the_log_record(self, tmp_path, caplog):
        apps = tmp_path / ".kirocrew" / "apps"
        forged = apps / "alpha\nSECURITY: forged"
        # A plain file at the ``data`` name is the per-app refusal branch, which logs.
        # The name contract refuses first, so that branch is never reached.
        forged.mkdir(parents=True)
        (forged / "data").write_text("x", encoding="utf-8")

        with caplog.at_level(logging.DEBUG, logger="kiro_crew.sandbox"):
            assert sandbox.app_data_window_targets(str(apps)) == ()

        for record in caplog.records:
            assert "\n" not in record.getMessage()
            assert "SECURITY: forged" not in record.getMessage()

    @_POSIX_ONLY
    def test_an_escape_in_an_app_name_does_not_reach_the_log_record(self, tmp_path, caplog):
        apps = tmp_path / ".kirocrew" / "apps"
        forged = apps / "alpha\x1b[2J"
        forged.mkdir(parents=True)
        (forged / "data").write_text("x", encoding="utf-8")

        with caplog.at_level(logging.DEBUG, logger="kiro_crew.sandbox"):
            assert sandbox.app_data_window_targets(str(apps)) == ()

        for record in caplog.records:
            assert "\x1b" not in record.getMessage()

    def test_every_log_site_that_names_an_entry_sanitizes_it(self):
        """The sanitization stays even though no admitted name can demonstrate it.

        Scoped to ``logger`` calls inside this one function: a bare ``entry.name`` in an
        argument list is the regression, and it cannot be caught by calling the function
        once the contract refuses every name that would prove it.
        """
        source = inspect.getsource(sandbox.app_data_window_targets)
        calls = re.findall(r"logger\.\w+\((?:[^()]|\([^()]*\))*\)", source, re.DOTALL)
        naming = [call for call in calls if "entry.name" in call]

        assert naming, "no log site names an entry, so this pin has nothing to protect"
        for call in naming:
            assert "safe_terminal_line(entry.name)" in call
            assert not re.search(r"(?<!safe_terminal_line\()entry\.name(?!\))", call)


class TestTheLauncherPinsAWindowAgainstASwappedAncestor:
    """The window is resolved a component at a time, from its mask entry down.

    ``O_NOFOLLOW`` on one open of the whole path refuses a link only at the LAST
    component, so a same-uid process that swaps an ANCESTOR -- making ``apps/alpha`` a
    link to ``apps/aws-control`` after the parent enumerated the windows -- would still
    have its target traversed, and the staging bind runs before the mask loop, so the
    masked leaf would be re-bound read-write at the window's path.

    Asserted over the generated launcher text, which is where the walk lives: the
    descriptor-relative form must be present AND the whole-path form must be absent.
    The second half is the one that catches a regression, since a reviewer adding a
    convenience open beside the walk leaves the first half true.
    """

    @_POSIX_ONLY
    def test_every_component_below_the_mask_is_opened_relative_to_a_descriptor(self):
        script = sandbox._build_launcher_script(
            "cc",
            extra_hidden_dirs=(os.path.join(os.path.expanduser("~"), ".kirocrew", "apps"),),
            extra_private_dirs=(
                os.path.join(os.path.expanduser("~"), ".kirocrew", "apps", "alpha", "data"),
            ),
        )

        assert "os.O_PATH | os.O_NOFOLLOW | os.O_DIRECTORY" in script
        assert "dir_fd=fd" in script
        # The regression shape: one open of the joined path, which leaves every
        # ancestor component following.
        assert "os.open(p, os.O_PATH" not in script

    @_POSIX_ONLY
    def test_the_bind_source_is_the_descriptor_not_the_name(self):
        """A second name resolution is a second chance to be redirected."""
        script = sandbox._build_launcher_script(
            "cc",
            extra_hidden_dirs=(os.path.join(os.path.expanduser("~"), ".kirocrew", "apps"),),
            extra_private_dirs=(
                os.path.join(os.path.expanduser("~"), ".kirocrew", "apps", "alpha", "data"),
            ),
        )

        assert "/proc/self/fd/%d" in script
        assert 'staging private window %s" % p' in script


class TestThePrivateWindowGate:
    """``_private_window_spellings`` admits a window only where one is safe.

    Both directions are pinned: a window that re-exposes a hidden target is refused, and
    an ordinary window inside the same masked tree is still admitted. Without the second
    assertion a gate that refused everything would pass the first.

    Paths are built from ``tmp_path`` rather than written as literals, because the rule is
    lexical over ``os.sep`` and a POSIX literal makes every case vacuous on Windows.
    """

    @staticmethod
    def _paths(tmp_path: Path) -> tuple[str, str, str, str]:
        apps = str(tmp_path / "apps")
        return (
            apps,
            os.path.join(apps, "aws-control", "data"),
            os.path.join(apps, "meetings", "data"),
            os.path.join(apps, "alpha", "data"),
        )

    def test_a_window_equal_to_a_hidden_target_is_refused(self, tmp_path):
        apps, leaf, _contains, _ordinary = self._paths(tmp_path)
        assert sandbox._private_window_spellings((leaf,), [apps, leaf]) == []

    def test_a_window_containing_a_hidden_target_is_refused(self, tmp_path):
        apps, _leaf, contains, _ordinary = self._paths(tmp_path)
        nested = os.path.join(contains, "edits")
        assert sandbox._private_window_spellings((contains,), [apps, nested]) == []

    def test_an_ordinary_window_inside_the_masked_tree_is_admitted(self, tmp_path):
        apps, leaf, _contains, ordinary = self._paths(tmp_path)
        assert sandbox._private_window_spellings((ordinary,), [apps, leaf]) == [ordinary]

    def test_a_window_outside_every_masked_tree_is_dropped(self, tmp_path):
        apps, _leaf, _contains, _ordinary = self._paths(tmp_path)
        outside = str(tmp_path / "elsewhere")
        assert sandbox._private_window_spellings((outside,), [apps]) == []

    def test_a_trailing_separator_does_not_defeat_the_refusal(self, tmp_path):
        """The spellings differ by a separator the comparison has to normalise."""
        apps, leaf, _contains, _ordinary = self._paths(tmp_path)
        assert sandbox._private_window_spellings((leaf,), [apps, leaf + os.sep]) == []


class TestTheCallerMaskMaterializer:
    @_POSIX_ONLY
    def test_it_creates_an_absent_target_owner_only_and_reports_it(self, tmp_path):
        target = tmp_path / "apps"
        assert materialize_caller_masked_dir(str(target)) is True
        assert target.is_dir()
        assert stat.S_IMODE(target.stat().st_mode) == 0o700

    def test_an_existing_directory_is_left_alone_and_not_reported(self, tmp_path):
        target = tmp_path / "apps"
        target.mkdir()
        marker = target / "installed"
        marker.mkdir()
        assert materialize_caller_masked_dir(str(target)) is False
        assert marker.is_dir()

    def test_a_plain_file_at_the_path_refuses(self, tmp_path):
        target = tmp_path / "apps"
        target.write_text("not a directory", encoding="utf-8")
        with pytest.raises(SandboxCeilingUnsealable):
            materialize_caller_masked_dir(str(target))

    def test_an_absent_parent_refuses_rather_than_building_the_chain(self, tmp_path):
        """No ``parents=True``: an intermediate directory created here would be an
        agent-writable ancestor a rename could swap out from under the mount."""
        target = tmp_path / "never" / "created" / "apps"
        with pytest.raises(SandboxCeilingUnsealable):
            materialize_caller_masked_dir(str(target))
        assert not (tmp_path / "never").exists()

    @_POSIX_ONLY
    def test_a_dangling_link_squatting_the_name_refuses(self, tmp_path):
        target = tmp_path / "apps"
        target.symlink_to(tmp_path / "gone")
        with pytest.raises(SandboxCeilingUnsealable):
            materialize_caller_masked_dir(str(target))


class TestWhatMaskingThisTreeStillCosts:
    @_POSIX_ONLY
    def test_a_write_outside_any_data_dir_is_discarded_rather_than_refused(self):
        """The residual, narrowed to the part of the tree that is not persistence.

        The Linux mask is a plain bind of an empty directory -- ``SENSITIVE_DIRS``, not
        ``READONLY_DIRS`` -- so a write under the tree inside the child SUCCEEDS and
        vanishes when the child exits. Each app's ``data`` root is carved back out as a
        window, so what remains is a write to an app's BUNDLE: its manifest, its code,
        its ``.app_secret``. None of those is a place a cron is meant to write, and the
        credential among them is the whole reason the tree is masked.

        Making the rest fail loudly instead needs a read-only mask primitive threaded
        through all three backends, which is a change to the sandbox's own vocabulary
        rather than to this call site.
        """
        apps = os.path.join(os.path.expanduser("~"), ".kirocrew", "apps")
        data = os.path.join(apps, "alpha", "data")
        script = sandbox._build_launcher_script(
            "cc", extra_hidden_dirs=(apps,), extra_private_dirs=(data,)
        )
        masked = json.loads(re.search(r"SENSITIVE_DIRS = (\[.*?\])\n", script, re.S).group(1))
        readonly = json.loads(re.search(r"READONLY_DIRS = (\[.*?\])\n", script, re.S).group(1))
        windows = json.loads(re.search(r"PRIVATE_DIRS = (\[.*?\])\n", script, re.S).group(1))

        assert apps in masked
        assert apps not in readonly
        # The bundle keeps the discard-on-exit behaviour; the data root does not.
        assert data in windows
