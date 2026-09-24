"""Which gates cover a write to the crew runtime config, and which do not.

``config.json`` holds the standing approval posture
(``agent.dangerously_skip_permissions``), so a process that can write it can turn
a session-scoped elevation into the default for every later session. This module
pins WHICH control refuses that write today, one assertion per control, so the
coverage is a measured fact rather than a reading of the tier lists.

The state these assertions describe is the current one. Two of the three controls
that could refuse a SHELL never reach a path: the read+write floor excludes these
leaves so that reading config stays routine, and the bash gate matches no paths at
all. What holds a shell writing the sealed NAME is the kernel: the OS sandbox seals
both leaves read-only, and pre-creates them so an install that has never saved
settings has a name for that seal to bind. That seal covers a path and not the inode
behind it, and the wide remedy for that -- refusing every spawn on a host whose
config carries a second name -- is deliberately not taken here. These assertions pin
the seal, the two gaps that make it the only cover, and the boundary it leaves.
"""

from __future__ import annotations

import pytest

from kiro_crew import sandbox
from kiro_crew.security import (
    is_sensitive_bash_command,
    is_sensitive_path,
    is_sensitive_write_path,
)

#: Both crew data-home spellings carry the same runtime config leaves.
CONFIG_LEAVES = ("config.json", "config.local.json")

HOME_PREFIX = "~/.kiro/crew"


class TestFileEditToolGate:
    """The agent's file-edit tool is refused, and the predicate discriminates."""

    def test_write_tier_covers_every_config_leaf(self) -> None:
        for leaf in CONFIG_LEAVES:
            assert is_sensitive_write_path(f"{HOME_PREFIX}/{leaf}") is True, leaf

    def test_write_tier_answers_false_for_an_ordinary_file(self) -> None:
        # Control: the tier is discriminating, so True above is a classification
        # rather than a predicate that accepts anything.
        assert is_sensitive_write_path(f"{HOME_PREFIX}/ordinary-note.txt") is False


class TestNoPathPredicateReachesTheShell:
    """Neither path predicate a shell could consult covers the config leaves.

    Each assertion records a deliberate gap, and together they are why the kernel
    seal below is the only control over a sandboxed shell's write.
    """

    def test_read_write_floor_does_not_cover_the_config_leaves(self) -> None:
        # Reading config is routine and intended, so the floor excludes these
        # leaves. The OS credential mask is projected from the floor, so a leaf
        # absent from the floor is absent from that mask.
        for leaf in CONFIG_LEAVES:
            assert is_sensitive_path(f"{HOME_PREFIX}/{leaf}") is False, leaf

    def test_shell_form_gate_does_not_refuse_a_redirect_onto_the_config(self) -> None:
        # The bash gate reads a command line for credential exfiltration and IMDS
        # reach; it matches no paths, so a redirect naming the config is clean to it.
        command = f"printf '{{}}' > {HOME_PREFIX}/config.json"
        assert is_sensitive_bash_command(command) is None


class TestSandboxSealsTheConfigLeaves:
    """The OS sandbox seals both leaves read-only and leaves reads working."""

    def test_both_config_leaves_are_sealed_read_only(self) -> None:
        sealed = set(sandbox._CREW_READONLY_LEAVES)
        for leaf in CONFIG_LEAVES:
            assert leaf in sealed, leaf

    def test_neither_config_leaf_is_masked(self) -> None:
        # In-sandbox readers resolve the subagent cap, the quarantine threshold and
        # the browser preference from this document, so the seal must deny writes
        # without hiding content.
        masked = set(sandbox._CREW_HIDDEN_LEAVES)
        for leaf in CONFIG_LEAVES:
            assert leaf not in masked, leaf

    def test_an_ordinary_leaf_is_not_sealed(self) -> None:
        # Control for both assertions above: membership is a classification rather
        # than a set that happens to contain everything.
        assert "ordinary-note.txt" not in set(sandbox._CREW_READONLY_LEAVES)


class TestSandboxGovernanceClassification:
    """A sealed leaf must be classified, and these two are withheld.

    ``test_sandbox_governance_mask`` pins the readable and withheld lists complete
    and disjoint over the sealed and visible sources, so sealing a leaf without
    classifying it cannot ship. Withheld is the side that fails safe here: an empty
    bind would read as no standing grant and no channel token.
    """

    def test_both_config_leaves_are_withheld_from_a_foreign_child(self) -> None:
        for leaf in CONFIG_LEAVES:
            assert leaf in set(sandbox._CREW_CHILD_WITHHELD_LEAVES), leaf

    def test_neither_config_leaf_is_child_readable(self) -> None:
        readable = set(sandbox.crew_host_runtime_leaves())
        for leaf in CONFIG_LEAVES:
            assert leaf not in readable, leaf


class TestTheAbsentDocumentBypassIsClosed:
    """A host that has never saved settings is covered too.

    ``mount(2)`` skips an absent path, and nothing writes these documents at startup,
    so on such a host the seal alone binds nothing and the overlay that wins the deep
    merge stays creatable from inside the sandbox. Pre-creation supplies the name the
    seal needs. An empty document is the safe content here: it resolves the standing
    approval posture to its default refusal.
    """

    def test_both_config_leaves_are_pre_created_so_the_seal_has_a_name(self) -> None:
        precreated = set(sandbox._CREW_PRECREATE_READONLY_FILE_LEAVES)
        for leaf in CONFIG_LEAVES:
            assert leaf in precreated, leaf

    def test_every_pre_created_leaf_is_also_sealed(self) -> None:
        # Control: pre-creation is only meaningful for a leaf the seal then binds, so
        # the list is a subset of the sealed set rather than an independent roster.
        sealed = set(sandbox._CREW_READONLY_LEAVES)
        assert set(sandbox._CREW_PRECREATE_READONLY_FILE_LEAVES) <= sealed

    def test_an_empty_document_resolves_the_posture_to_refusal(self) -> None:
        # The content criterion pre-creation rests on: what a pre-created ``{}`` means
        # is "no standing grant", so a sandbox pinned at it reads the safe value.
        from kiro_crew.config.loader import KiroCrewConfig

        assert KiroCrewConfig().agent.dangerously_skip_permissions is False


class TestTheStandingGrantMovesToAnUnopenableLeaf:
    """The switch itself moves, which is the only placement that closes the residual.

    A read-only seal covers a PATH; a mask covers the NAME for every present and future
    spelling, so an agent cannot open the document at all -- it can neither read the grant
    nor obtain a `link(2)` source for it. That placement is available for this leaf and not
    for `config.json` for one measurable reason: nothing in-sandbox reads it.

    Both directions are asserted, because either one alone would pass a broken change: a
    leaf nobody can write is useless if the operator cannot grant, and a leaf the operator
    can grant on is useless if the agent can write it too.
    """

    LEAF = "standing_approval.json"

    def test_the_leaf_is_masked_not_merely_sealed(self) -> None:
        # Masked is the property that makes the name unopenable. Read-only would leave the
        # document readable, which is also what makes it a link source.
        assert self.LEAF in set(sandbox._CREW_HIDDEN_LEAVES)
        assert self.LEAF not in set(sandbox._CREW_READONLY_LEAVES)

    def test_the_leaf_is_on_the_read_and_write_floor(self) -> None:
        # The tool path, which is the tier the mask does not cover.
        assert is_sensitive_path(f"{HOME_PREFIX}/{self.LEAF}") is True
        assert is_sensitive_write_path(f"{HOME_PREFIX}/{self.LEAF}") is True

    def test_the_staging_directory_is_masked_and_pre_created(self) -> None:
        # The temp staged there BECOMES the grant document, so a visible temp name would
        # be a writable second path to it.
        staging = "standing-approval-staging"
        assert staging in set(sandbox._CREW_HIDDEN_LEAVES)
        assert staging in set(sandbox._CREW_PRECREATE_HIDDEN_DIR_LEAVES)
        assert is_sensitive_path(f"{HOME_PREFIX}/{staging}") is True

    def test_a_materialiser_gives_the_mask_a_name_to_bind(self) -> None:
        # Absent is the DEFAULT state for this leaf, and a mask skips an absent target, so
        # without this the name stays creatable and an agent self-grants.
        assert callable(sandbox._materialize_standing_approval_mask_target)

    def test_the_operator_read_path_answers_a_real_grant(self, tmp_path, monkeypatch) -> None:
        from kiro_crew import safety_override as so_mod

        target = tmp_path / self.LEAF
        target.write_text('{"enabled": true}', encoding="utf-8")
        monkeypatch.setattr(so_mod, "standing_approval_path", lambda: target)

        assert so_mod.standing_grant_declared() is True

    @pytest.mark.parametrize(
        "body",
        (
            '{"enabled": false}',
            "{}",
            '{"enabled": "true"}',
            '{"enabled": 1}',
            "[]",
            "not json at all",
            "",
        ),
    )
    def test_every_other_document_answers_no_grant(self, tmp_path, monkeypatch, body) -> None:
        # Fails soft in every direction, and `true` is required exactly so a truthy string
        # or a non-zero number cannot become an authorization by accident.
        from kiro_crew import safety_override as so_mod

        target = tmp_path / self.LEAF
        target.write_text(body, encoding="utf-8")
        monkeypatch.setattr(so_mod, "standing_approval_path", lambda: target)

        assert so_mod.standing_grant_declared() is False

    def test_an_absent_document_answers_no_grant(self, tmp_path, monkeypatch) -> None:
        from kiro_crew import safety_override as so_mod

        monkeypatch.setattr(so_mod, "standing_approval_path", lambda: tmp_path / self.LEAF)

        assert so_mod.standing_grant_declared() is False

    def test_the_precreated_stub_is_absent_equivalent(self, tmp_path, monkeypatch) -> None:
        # The argument the pre-create list asks each masked file leaf to supply: what the
        # sandbox is pinned at must read the same as no file at all.
        from kiro_crew import safety_override as so_mod

        target = tmp_path / self.LEAF
        target.write_bytes(sandbox._STANDING_APPROVAL_PRECREATE_CONTENT)
        monkeypatch.setattr(so_mod, "standing_approval_path", lambda: target)

        assert so_mod.standing_grant_declared() is False

    def test_the_config_key_alone_no_longer_grants(self, tmp_path, monkeypatch) -> None:
        # The migration's teeth: an operator who has only the old key gets no grant, and
        # the warning names the file to write.
        from kiro_crew import safety_override as so_mod

        target = tmp_path / self.LEAF
        monkeypatch.setattr(so_mod, "standing_approval_path", lambda: target)

        assert so_mod.standing_grant_declared() is False
        so_mod.warn_if_config_declares_standing_grant(True)


class TestWhatANameBasedSealDoesNotCover:
    """The boundary of the seal, asserted rather than assumed.

    The seal is name-based on every platform it exists on: Linux binds a MOUNT read-only
    and macOS denies `file-write*` on a pathname. A bind covers a PATH, not the inode
    behind it, and the crew data-home root is writable in every sandbox, so a process
    that owns the document can add a second name to that inode and write through it. The
    module records the same shape for the live-target pointer. Closing it means making the
    document unopenable in-sandbox, which these leaves cannot be: the loader reads them
    per call. The standing-approval switch's own move onto the read-gate floor is the
    shape that closes it, and it is tracked separately.

    What this class pins is that the wide remedy is NOT taken here, because that choice is
    the one a review has blocked before and it must not drift in silently.
    """

    def test_neither_config_leaf_is_on_the_spawn_refusal_list(self) -> None:
        # A leaf on this list refuses every sandboxed spawn on the host -- every chat
        # turn, cron job and subagent -- once the leaf carries a second name, which
        # ordinary dotfile tooling leaves behind. That is the whole box for one grant's
        # exposure, and it belongs to the operator rather than to this change.
        spawn_refused = set(sandbox._CREW_NOFOLLOW_READONLY_FILE_LEAVES)
        for leaf in CONFIG_LEAVES:
            assert leaf not in spawn_refused, leaf

    def test_the_strict_alias_check_has_no_config_leaf_caller(self) -> None:
        # The complement: no seam quietly claims to answer the alias question for these
        # leaves, so the residual above is a declared boundary rather than a gap hidden
        # behind a check that cannot hold.
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(sandbox))
        callers = {
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef)
            for c in ast.walk(fn)
            if isinstance(c, ast.Call) and ast.unparse(c.func) == "_require_real_file_nofollow"
        }
        assert callers == {
            "require_unaliased_cloud_config",
            "require_unaliased_launch_state",
        }, sorted(callers)
