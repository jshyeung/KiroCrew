"""The standing auto-approve switch lives on an UNOPENABLE keystone, not in config.json.

A read-only seal on ``config.json`` closes a write to the sealed NAME. It cannot close
the inode behind that name: the crew data-home root is writable in every sandbox, the
agent runs as the operator's uid and so OWNS the document, ``link(2)`` needs no write
permission on the file it names a second time, and a bind mount seals a MOUNT rather
than an inode. ``sandbox`` states that fact itself where it refuses a governance
ceiling whose ``st_nlink`` is not 1.

The switch therefore lives on a leaf a sandboxed process cannot OPEN. Two properties
carry that, and each gets its own assertion below because either alone is
insufficient:

* masked rather than sealed read-only -- a sealed-but-readable leaf is still a
  ``link(2)`` source;
* a DIRECTORY rather than a file -- Linux refuses ``link(2)`` on a directory outright,
  so the alias shape has no source even in principle.

Every claim has a discriminating control beside it, so a vacuous or misspelled set
cannot make this file pass.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from kiro_crew import sandbox, standing_approval
from kiro_crew.config import loader
from kiro_crew.security import paths as security_paths


@pytest.fixture()
def crew_home(tmp_path, monkeypatch):
    """Point every reader of the data home at a scratch tree."""
    home = tmp_path / ".kiro" / "crew"
    home.mkdir(parents=True)
    monkeypatch.setattr(loader, "config_dir", lambda: home)
    monkeypatch.setattr(sandbox, "config_dir", lambda: home)
    return home


def _write_grant(home, document: object) -> None:
    leaf = home / loader.STANDING_APPROVAL_DIRNAME
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / loader.STANDING_APPROVAL_FILENAME).write_text(
        document if isinstance(document, str) else json.dumps(document),
        encoding="utf-8",
    )


class TestKeystonePlacement:
    """Where the leaf sits in the two fences, and that the two agree on its name."""

    def test_the_leaf_is_on_the_agent_file_tool_keystone_floor(self):
        assert loader.STANDING_APPROVAL_DIRNAME in security_paths._CREW_SECRET_LEAVES
        # Discriminating control: an ordinary crew-home leaf is NOT on the floor, so a
        # pass above cannot come from a set that swallows every name.
        assert "config.json" not in security_paths._CREW_SECRET_LEAVES

    def test_the_leaf_is_masked_not_merely_sealed_read_only(self):
        """The whole point of the move: unopenable in-sandbox, not write-denied.

        A read-only seal leaves the document READABLE, and a readable inode the caller
        owns is a ``link(2)`` source. Being in the readonly set instead of the hidden
        one would reproduce exactly the residual this leaf exists to close.
        """
        assert loader.STANDING_APPROVAL_DIRNAME in sandbox._CREW_HIDDEN_LEAVES
        assert loader.STANDING_APPROVAL_DIRNAME not in sandbox._CREW_READONLY_LEAVES
        # Control: the sealed-but-readable disposition really is a different set with
        # real members, so the second assertion is not vacuously true.
        assert "computer_use.json" in sandbox._CREW_READONLY_LEAVES
        assert "computer_use.json" not in sandbox._CREW_HIDDEN_LEAVES

    def test_the_leaf_is_precreated_so_the_isdir_guarded_mask_is_not_vacuous(self):
        """An absent name gets NO bind, and absent is the default on every install.

        ``mount(2)`` cannot mask a path that does not exist and the mask loop guards on
        ``isdir``, so without pre-creation a sandboxed process could CREATE the
        directory in the writable data-home root and write the grant the gateway reads
        back as the operator's own standing authority.
        """
        assert loader.STANDING_APPROVAL_DIRNAME in sandbox._CREW_PRECREATE_HIDDEN_DIR_LEAVES

    def test_the_mask_and_the_reader_name_the_same_directory(self):
        """``sandbox`` spells the leaf as a literal to stay off the config-loader import
        chain, exactly as it does for the live-target pointer. This is the pin that
        keeps the two spellings from drifting into two different directories -- which
        would mask one name while the gateway read another.
        """
        assert sandbox._STANDING_APPROVAL_LEAF == loader.STANDING_APPROVAL_DIRNAME

    def test_the_grant_document_lives_inside_the_classified_directory(self, crew_home):
        """A directory entry covers every child; a file entry would leave the container
        writable, which is the same hole one level up.
        """
        path = loader.standing_approval_path()
        assert path.parent.name == loader.STANDING_APPROVAL_DIRNAME
        assert path.name == loader.STANDING_APPROVAL_FILENAME
        assert path.parent.parent == crew_home

    def test_a_hidden_leaf_needs_no_child_readable_classification(self):
        """``test_sandbox_governance_mask`` pins the child-readable/withheld pair
        complete and disjoint over ``_CREW_SANDBOX_VISIBLE_LEAVES |
        _CREW_READONLY_LEAVES``. A hidden leaf is in neither source, so it is correctly
        absent from both halves rather than unclassified.
        """
        union = set(sandbox._CREW_SANDBOX_VISIBLE_LEAVES) | set(sandbox._CREW_READONLY_LEAVES)
        assert loader.STANDING_APPROVAL_DIRNAME not in union
        assert loader.STANDING_APPROVAL_DIRNAME not in sandbox._CREW_CHILD_READABLE_LEAVES
        assert loader.STANDING_APPROVAL_DIRNAME not in sandbox._CREW_CHILD_WITHHELD_LEAVES


class TestPrecreateIsBehavioural:
    """Membership in the precreate list is not the property; creation is."""

    def test_materialise_creates_the_directory_owner_only(self, crew_home):
        target = crew_home / loader.STANDING_APPROVAL_DIRNAME
        assert not target.exists()  # fresh install: nothing declared yet

        created = sandbox._materialize_maskable_dirs()

        assert target.is_dir()
        assert str(target) in created
        if os.name == "posix":
            # Owner-only whatever the umask: no group or other access.
            assert (target.stat().st_mode & 0o077) == 0

    def test_an_empty_directory_is_the_readers_absent_equivalent(self, crew_home):
        """What a sandboxed reader would see through the mask must mean NO GRANT.

        This is the criterion the precreate lists require of every leaf they
        materialise, and for a GRANT the empty form is the safe direction.
        """
        sandbox._materialize_maskable_dirs()
        assert (crew_home / loader.STANDING_APPROVAL_DIRNAME).is_dir()
        assert standing_approval.is_declared() is False


class TestDirectoryHasNoAliasSource:
    """The property that makes a directory close the hole rather than move it."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX link(2) semantics")
    def test_the_kernel_refuses_a_second_name_for_a_directory(self, crew_home):
        """``link(2)`` on a directory is EPERM, so the shape the issue reports for a
        sealed FILE has no source here. Run against the real keystone directory rather
        than an arbitrary one, so the assertion is about this leaf.
        """
        sandbox._materialize_maskable_dirs()
        target = crew_home / loader.STANDING_APPROVAL_DIRNAME
        alias = crew_home / "alias-attempt"

        with pytest.raises(OSError) as err:
            os.link(str(target), str(alias), follow_symlinks=False)

        assert not alias.exists()
        # Control: the same call on a regular FILE in the same directory SUCCEEDS --
        # which is the residual for a sealed file leaf, and why this leaf is a
        # directory.
        regular = crew_home / "ordinary.json"
        regular.write_text("{}", encoding="utf-8")
        regular.chmod(0o444)
        file_alias = crew_home / "ordinary-alias.json"
        os.link(str(regular), str(file_alias))
        assert file_alias.stat().st_ino == regular.stat().st_ino
        assert err.value.errno != 0


class TestReadsFailClosed:
    """An absent or unreadable leaf resolves to refusal -- the issue's done-when."""

    def test_absent_leaf_is_no_grant(self, crew_home):
        assert standing_approval.is_declared() is False

    def test_absent_document_inside_an_existing_directory_is_no_grant(self, crew_home):
        (crew_home / loader.STANDING_APPROVAL_DIRNAME).mkdir()
        assert standing_approval.is_declared() is False

    def test_unparseable_document_is_no_grant(self, crew_home):
        _write_grant(crew_home, "{not json")
        assert standing_approval.is_declared() is False

    def test_a_json_non_object_is_no_grant(self, crew_home):
        _write_grant(crew_home, [True])
        assert standing_approval.is_declared() is False

    @pytest.mark.parametrize("value", ["true", "false", "0", "no", 1, 0, [], {}, None])
    def test_only_a_real_boolean_true_grants(self, crew_home, value):
        """A truthy STRING must not grant. ``"false"`` and ``"0"`` are truthy in
        Python, so a bare ``bool(...)`` here would read an explicit disable as the
        standing grant -- the same trap ``_read_skip_permissions`` documents.
        """
        _write_grant(crew_home, {standing_approval.GRANT_FIELD: value})
        assert standing_approval.is_declared() is False

    def test_an_explicit_boolean_true_grants(self, crew_home):
        _write_grant(crew_home, {standing_approval.GRANT_FIELD: True})
        assert standing_approval.is_declared() is True

    def test_an_explicit_boolean_false_does_not_grant(self, crew_home):
        _write_grant(crew_home, {standing_approval.GRANT_FIELD: False})
        assert standing_approval.is_declared() is False

    def test_an_unreadable_document_is_no_grant(self, crew_home):
        if os.name != "posix" or os.geteuid() == 0:
            pytest.skip("needs POSIX permission bits and a non-root uid")
        _write_grant(crew_home, {standing_approval.GRANT_FIELD: True})
        path = loader.standing_approval_path()
        path.chmod(0o000)
        try:
            assert standing_approval.is_declared() is False
        finally:
            path.chmod(0o600)
        # Control: the very same document reads as a grant once it is readable again,
        # so the refusal above came from the permission and not from the content.
        assert standing_approval.is_declared() is True


class TestConfigKeyNoLongerGrants:
    """The migration is explicit: the retired key grants nothing and says so."""

    def test_the_retired_key_is_not_read_by_the_keystone_reader(self, crew_home):
        """The reader must not consult ``config.json`` at all. Writing the retired
        declaration into a config document next to an absent keystone must leave the
        answer at no-grant.
        """
        (crew_home / "config.json").write_text(
            json.dumps({"agent": {"dangerously_skip_permissions": True}}), encoding="utf-8"
        )
        assert standing_approval.is_declared() is False

    def test_the_migration_notice_names_the_path_and_the_document(self):
        notice = standing_approval.migration_notice()
        assert loader.STANDING_APPROVAL_DIRNAME in notice
        assert loader.STANDING_APPROVAL_FILENAME in notice
        assert "dangerously_skip_permissions" in notice
        # Actionable for a reader with no context: it must say the old key grants
        # nothing, not merely that something moved.
        assert "no longer grants" in notice

    def test_the_notice_is_ascii_so_every_log_sink_renders_it(self):
        standing_approval.migration_notice().encode("ascii")


class TestStartupHonoursOnlyTheKeystone:
    """The gateway startup path reads the keystone, and announces a stranded key."""

    def _cfg(self, declared: bool):
        class _Agent:
            dangerously_skip_permissions = declared

        class _Cfg:
            agent = _Agent()

        return _Cfg()

    @pytest.fixture()
    def startup(self, monkeypatch, crew_home):
        from kiro_crew.dashboard import server

        monkeypatch.setattr(server, "apply_config_duration", lambda: None)
        calls: list[str] = []

        class _Result:
            active = True
            ttl = 0

        monkeypatch.setattr(
            server, "grant_declared_yolo", lambda: (calls.append("granted"), _Result())[1]
        )
        return server, calls

    def test_a_stranded_config_key_grants_nothing(self, startup, caplog):
        server, calls = startup
        with caplog.at_level("WARNING"):
            server._apply_startup_yolo(object(), self._cfg(declared=True))
        assert calls == []
        assert "no longer grants" in caplog.text

    def test_the_keystone_grants_and_logs_no_migration_warning(self, startup, caplog, crew_home):
        server, calls = startup
        _write_grant(crew_home, {standing_approval.GRANT_FIELD: True})
        with caplog.at_level("WARNING"):
            server._apply_startup_yolo(object(), self._cfg(declared=False))
        assert calls == ["granted"]
        assert "no longer grants" not in caplog.text

    def test_neither_set_grants_nothing_and_says_nothing(self, startup, caplog):
        server, calls = startup
        with caplog.at_level("WARNING"):
            server._apply_startup_yolo(object(), self._cfg(declared=False))
        assert calls == []
        assert "no longer grants" not in caplog.text
