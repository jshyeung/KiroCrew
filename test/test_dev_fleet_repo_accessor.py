"""``MAIN_REPO`` reaches git and the filesystem only through ``_repo()``.

``MAIN_REPO`` reaches git and the filesystem only through the accessors.

Dev Fleet represents "no main checkout found" as an empty string in
``MAIN_REPO``. That sentinel is fail-open at any call site that consumes the
global directly: ``git -C ""`` does not fail — it silently runs against the
backend process's working directory — and ``Path("")`` is ``Path(".")``, so an
unguarded consumer operates on an arbitrary directory and returns plausible
results. ``_repo_read()`` centralizes the guard: it returns the path or
raises ``RepoNotConfigured``, which the HMAC middleware converts to the 409
``repo_not_configured`` boundary. ``_repo()`` is the MUTATING accessor and adds
one refusal on top — a checkout served read-only — and it reaches the path
through ``_repo_read()``, so exactly one function still reads the global.

Two enforcement tiers (same pattern as ``test_apps_instances_loop_offload.py``):

- Behavior tests: ``_repo_read()`` raises on the empty sentinel and returns the
  path otherwise; ``_repo()`` additionally raises ``RepoReadOnly`` while the
  read-only state is set. Both preserve the exception types the middleware
  boundary maps.
- AST ratchet: outside the read accessor itself, a ``MAIN_REPO`` load may appear
  ONLY as a bare truthiness guard (``if MAIN_REPO:`` / ``not MAIN_REPO`` / a
  ``BoolOp`` operand). Any other load — a git argv element, a subprocess
  ``cwd=``, a ``Path(...)`` build, an f-string interpolation, a payload
  field — fails this test, so a future call site cannot silently reintroduce
  the fail-open shape.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import locale
from types import SimpleNamespace

import pytest

from kiro_crew.apps.builtins.dev_fleet import (
    fleet_state,
    gateway_routes,
    http_api,
    live,
    repository,
    runtime,
    server,
    worktree_ops,
)

# The read accessor is the ONLY function whose body may read the bare global: it
# IS the guard, and the mutating accessor delegates to it rather than loading the
# global a second time, so the count of functions touching MAIN_REPO stays at one.
# The startup hook's discovery/re-resolve runs on a local and writes the global
# exactly once (a Store, which this ratchet ignores), so even the assignment site
# needs no exemption — and a git call added to startup, where MAIN_REPO is most
# often still unresolved, is caught like anywhere else.
_DEV_FLEET_MODULES = (
    runtime,
    repository,
    live,
    fleet_state,
    worktree_ops,
    http_api,
    server,
)
_ALLOWED_LOADS = {(repository.__name__, "_repo_read")}


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parents.get(cur)
    return None


def _is_bare_truthiness(node: ast.expr, parents: dict[ast.AST, ast.AST]) -> bool:
    """True when the load feeds a truthiness test and nothing else.

    Walking up from the Name, only ``BoolOp`` and ``not`` may intervene before
    the expression lands as the ``test`` of an ``if``/``while`` or a ternary.
    Any other intervening node (a call argument, a container literal, an
    f-string, an assignment value) means the VALUE escapes, which is exactly
    the shape the accessor exists to prevent.
    """
    child: ast.AST = node
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.BoolOp, ast.UnaryOp)):
            if isinstance(cur, ast.UnaryOp) and not isinstance(cur.op, ast.Not):
                return False
            child = cur
            cur = parents.get(cur)
            continue
        if isinstance(cur, (ast.If, ast.While)):
            return cur.test is child
        if isinstance(cur, ast.IfExp):
            return cur.test is child
        return False
    return False


def test_main_repo_loads_only_via_accessor_or_truthiness() -> None:
    violations: list[str] = []
    for module in _DEV_FLEET_MODULES:
        tree = ast.parse(inspect.getsource(module))
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            is_main_repo = (isinstance(node, ast.Name) and node.id == "MAIN_REPO") or (
                isinstance(node, ast.Attribute) and node.attr == "MAIN_REPO"
            )
            if not is_main_repo or not isinstance(node.ctx, ast.Load):
                continue  # assignments (Store) stay on the global by design
            func = _enclosing_function(node, parents)
            if (module.__name__, func) in _ALLOWED_LOADS:
                continue
            if _is_bare_truthiness(node, parents):
                continue
            violations.append(
                f"{module.__name__}:{node.lineno}: MAIN_REPO load in "
                f"{func or '<module>'} — route it through repository._repo()"
                " (or _repo_read() for a read-only consumer)"
            )
    assert not violations, (
        "MAIN_REPO's empty-string sentinel is fail-open when consumed "
        "directly (git -C '' runs against the process CWD). Use _repo():\n" + "\n".join(violations)
    )


def test_repo_accessor_raises_on_unresolved_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repository, "MAIN_REPO", "")
    with pytest.raises(repository.RepoNotConfigured):
        repository._repo()


def test_repo_accessor_returns_resolved_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repository, "MAIN_REPO", "/somewhere/kirocrew")
    assert repository._repo() == "/somewhere/kirocrew"


def test_read_accessor_raises_on_unresolved_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The weaker accessor still refuses the fail-open sentinel.

    It is the one the ratchet admits, so if it ever stopped raising here the
    ratchet would be guarding a function that hands out ``""``.
    """
    monkeypatch.setattr(repository, "MAIN_REPO", "")
    with pytest.raises(repository.RepoNotConfigured):
        repository._repo_read()


def test_read_accessor_serves_a_checkout_the_app_may_only_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read-only is the whole point of the split: the generic surface still works."""
    monkeypatch.setattr(repository, "MAIN_REPO", "/somewhere/other-project")
    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", "read-only: no markers")
    assert repository._repo_read() == "/somewhere/other-project"


def test_mutating_accessor_refuses_a_checkout_the_app_may_only_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal every mutating call site inherits without being touched.

    ``RepoReadOnly`` must keep sharing the ``RepoUnavailable`` base, because the
    degrade sites catch that base and their "not derivable" answer is the right
    one here too.
    """
    monkeypatch.setattr(repository, "MAIN_REPO", "/somewhere/other-project")
    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", "read-only: no markers")
    with pytest.raises(repository.RepoReadOnly) as caught:
        repository._repo()
    assert isinstance(caught.value, repository.RepoUnavailable)
    assert "read-only" in str(caught.value)


def test_mutating_accessor_allows_a_marker_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate does not fire when the state is genuinely empty."""
    monkeypatch.setattr(repository, "MAIN_REPO", "/somewhere/kirocrew")
    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", None)
    assert repository._repo() == "/somewhere/kirocrew"


def test_read_only_reason_reports_the_state_both_ways(monkeypatch: pytest.MonkeyPatch) -> None:
    """One reader for the route boundary, the payload and the row fields."""
    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", None)
    assert repository._read_only_reason() is None
    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", "read-only: no markers")
    assert repository._read_only_reason() == "read-only: no markers"


@pytest.mark.parametrize(
    "name, ok",
    [
        ("main", True),
        ("trunk", True),
        ("release/2.0", True),
        ("feature.x", True),
        # A leading dash is parsed as a FLAG by git once the name is
        # interpolated into an argv, so it must never be accepted.
        ("--exec=touch /tmp/pwn", False),
        ("-main", False),
        # ``..`` splits a rev range at the wrong place: ``origin/a..b..HEAD``.
        ("a..b", False),
        ("", False),
        ("main branch", False),
        ("main;rm", False),
    ],
)
def test_base_branch_names_are_constrained_before_reaching_an_argv(name: str, ok: bool) -> None:
    assert repository._plausible_branch_name(name) is ok


def test_every_local_base_candidate_survives_the_argv_constraint() -> None:
    """The fallback list and the argv guard must agree.

    A candidate the guard rejects would be published into ``BASE_BRANCH`` by the
    fallback loop without ever meeting ``_plausible_branch_name``, which only
    screens the remote's answer. Asserting over the tuple itself keeps a name
    added later from slipping past.
    """
    assert repository._LOCAL_BASE_CANDIDATES
    for candidate in repository._LOCAL_BASE_CANDIDATES:
        assert repository._plausible_branch_name(candidate) is True


def test_primary_checkout_resolution_preserves_the_host_text_decoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Moving the startup probe must not reinterpret non-ASCII checkout paths."""
    primary = tmp_path / "primary"
    seen: dict[str, object] = {}

    def _run(_argv, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=str(primary / ".git"))

    monkeypatch.setattr(runtime, "_trusted_bin", lambda _name: "git")
    monkeypatch.setattr(repository.subprocess, "run", _run)

    assert repository._resolve_primary_checkout(str(tmp_path / "linked")) == str(primary)
    assert seen["text"] is True
    assert seen["encoding"] == locale.getpreferredencoding(False)


# --- base-branch resolution reads ONE remote ---------------------------------
#
# ``git remote`` lists names alphabetically, so a checkout carrying an archive or
# fork remote beside ``origin`` hands the first-listed one the casting vote. That
# remote decides ``BASE_BRANCH`` while ``_upstream_remote`` resolves to ``origin``
# independently, and ``/rebase`` rewrites onto ``{remote}/{BASE_BRANCH}`` — a base
# the upstream never published. These three pin which remote is consulted.


def _stub_base_branch_git(
    monkeypatch: pytest.MonkeyPatch, *, remotes: str, published: dict[str, str], local: set[str]
) -> list[str]:
    """Wire the two git readers ``_resolve_base_branch`` uses. Returns the ref probes."""
    probed: list[str] = []

    async def _run_cmd(argv, **_kwargs):
        assert argv[-1] == "remote", argv
        return 0, remotes, ""

    async def _git(_repo: str, *args: str) -> str | None:
        if args[0] == "symbolic-ref":
            ref = args[-1]
            probed.append(ref)
            remote = ref.split("/")[2]
            head = published.get(remote)
            return f"{remote}/{head}" if head else None
        if args[0] == "rev-parse":
            name = args[-1].removeprefix("refs/heads/")
            return name if name in local else None
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(repository, "MAIN_REPO", "/somewhere/other-project")
    monkeypatch.setattr(repository, "_REPO_INVALID_MSG", None)
    monkeypatch.setattr(repository, "BASE_BRANCH", "main")
    monkeypatch.setattr(runtime, "_run_cmd", _run_cmd)
    monkeypatch.setattr(repository, "_git", _git)
    return probed


@pytest.mark.asyncio
async def test_base_branch_ignores_a_remote_sorted_before_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alphabetically earlier remote must not decide the rebase base.

    ``archive`` publishes one default and ``origin`` another. Only ``origin`` may
    be consulted, because ``_upstream_remote`` resolves to it and the two answers
    are combined into a single rev range.
    """
    probed = _stub_base_branch_git(
        monkeypatch,
        remotes="archive\norigin\n",
        published={"archive": "legacy-default", "origin": "trunk"},
        local=set(),
    )
    await repository._resolve_base_branch()
    assert repository.BASE_BRANCH == "trunk"
    assert probed == ["refs/remotes/origin/HEAD"]


@pytest.mark.asyncio
async def test_base_branch_reads_a_sole_remote_under_another_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One remote is unambiguous whatever it is called, so its answer is taken."""
    probed = _stub_base_branch_git(
        monkeypatch,
        remotes="kirocrew\n",
        published={"kirocrew": "release/3"},
        local=set(),
    )
    await repository._resolve_base_branch()
    assert repository.BASE_BRANCH == "release/3"
    assert probed == ["refs/remotes/kirocrew/HEAD"]


@pytest.mark.asyncio
async def test_base_branch_falls_back_locally_when_no_remote_is_unambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several remotes and no ``origin`` is ambiguous: ask the local branches."""
    local_default = repository._LOCAL_BASE_CANDIDATES[-1]
    probed = _stub_base_branch_git(
        monkeypatch,
        remotes="fork\nupstream\n",
        published={"fork": "a", "upstream": "b"},
        local={local_default},
    )
    await repository._resolve_base_branch()
    assert repository.BASE_BRANCH == local_default
    assert probed == []


# --- reads leave the repository byte-identical -------------------------------


def test_optional_locks_are_off_for_every_git_this_handler_runs() -> None:
    """``git status`` rewrites the index unless optional locks are off.

    It is a read to its caller and a write to the repository: it refreshes the
    index's stat cache and saves it back under ``index.lock``. Every fleet render
    runs one per row, so without this a checkout the app may only read is modified
    on its ordinary path. Pinned on the env chokepoint rather than per call site,
    which is what makes a read added later inherit it.
    """
    assert runtime._GIT_ENV_NEUTRALIZERS["GIT_OPTIONAL_LOCKS"] == "0"


@pytest.mark.asyncio
async def test_run_cmd_puts_the_neutralizers_in_the_child_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dict is only a guarantee if the spawn actually carries it.

    Asserted through the spawn preparation, because that is the last place the env
    can be read before the child exists, and an entry dropped anywhere earlier
    would leave the dict stating a pin nothing applies.
    """
    seen: dict[str, str] = {}

    def _prepare(cmd, _mode, env=None):
        seen.update(env or {})
        return list(cmd), dict(env or {}), None

    monkeypatch.setattr(runtime, "sandboxed_spawn_argv", _prepare)
    monkeypatch.setattr(runtime, "_trusted_bin", lambda _name: "/usr/bin/git")

    async def _off_loop(fn, executor=None):
        return fn()

    monkeypatch.setattr(runtime, "shielded_prepare_off_loop", _off_loop)

    async def _no_child(*_a, **_kw):
        raise AssertionError("the env is read before the child spawns")

    monkeypatch.setattr(runtime.asyncio, "create_subprocess_exec", _no_child)
    with pytest.raises(AssertionError):
        await runtime._run_cmd(["git", "-C", "/somewhere/other-project", "status"])

    for key, value in runtime._GIT_ENV_NEUTRALIZERS.items():
        assert seen[key] == value


# --- the gateway's own boundary carries the refusal --------------------------


@pytest.mark.asyncio
async def test_gateway_repo_resolution_refuses_a_read_only_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway routes skip the backend middleware, so the refusal lives here.

    Make Live is why it matters: it reaches ``_find_worktree_by_path`` and writes
    the live-target pointer, which would aim the running gateway at a checkout
    this app may only read.
    """

    async def _discovered() -> None:
        return None

    monkeypatch.setattr(repository, "ensure_main_repo_discovered", _discovered)
    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", "read-only: no markers")

    refused = await gateway_routes._ensure_repo()
    assert refused is not None
    assert refused.status == 409
    assert json.loads(refused.body)["code"] == "repo_read_only"

    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", None)
    assert await gateway_routes._ensure_repo() is None


# --- a fenced path is refused, never served read-only ------------------------


def test_read_only_adoption_asks_the_central_path_gate() -> None:
    """The adoption branch must consult the gate, off the loop, on BOTH spellings.

    ``sensitive_path_refusal`` waits on the bounded path-resolution pool, so calling
    it in the coroutine body pauses every other task on the gateway loop. It also
    has to see the path the operator NAMED: ``_resolve_primary_checkout`` rewrites a
    linked worktree to its primary, and a fenced worktree can have a primary outside
    the fence, so gating only the rewritten form clears it by a path that is not it.
    """
    body = inspect.getsource(repository.ensure_main_repo_discovered)
    assert "sensitive_path_refusal" not in body, "the gate must not run in the coroutine body"
    assert "_fenced_reason(configured, discovered)" in body
    gate_call = body.index("_fenced_reason(configured, discovered)")
    adopt = body.index("read_only_msg = (")
    assert gate_call < adopt, "the gate must be consulted before the adoption message is built"

    probe = inspect.getsource(repository._fenced_reason)
    assert "for candidate in (configured, resolved)" in probe


def test_a_fenced_path_is_refused_outright(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fenced verdict publishes an unreadable refusal, never a read-only fleet."""
    monkeypatch.setattr(repository, "MAIN_REPO", "/somewhere/fenced")
    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", None)
    monkeypatch.setattr(repository, "_REPO_INVALID_MSG", "refused: protected location")

    assert repository._read_only_reason() is None
    with pytest.raises(repository.RepoUnreadable):
        repository._repo_read()


def test_both_path_spellings_reach_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The named path and the primary it normalizes to are asked separately."""
    asked: list[str] = []

    def _refusal(path: str, base_dir=None):
        asked.append(path)
        return "fenced" if path == "/named/linked-worktree" else None

    monkeypatch.setattr(repository, "sensitive_path_refusal", _refusal)
    assert repository._fenced_reason("/named/linked-worktree", "/other/primary") == "fenced"
    assert asked == ["/named/linked-worktree"]

    asked.clear()
    assert repository._fenced_reason("/clean/named", "/clean/primary") is None
    assert asked == ["/clean/named", "/clean/primary"]


def test_an_unverified_gate_answer_does_not_latch() -> None:
    """A resolver stall is a measurement nobody took, so the next poll retries.

    Latched, one transient timeout would stand as a permanent refusal asserting a
    verdict that was never reached, and the retry the gate itself advises could
    never fire.
    """
    body = inspect.getsource(repository.ensure_main_repo_discovered)
    assert "is_unverifiable_path_refusal(fenced)" in body
    assert "_DISCOVERY_DONE = bool(discovered) and not unverified" in body
    stall = body.index("is_unverifiable_path_refusal(fenced)")
    fence = body.index("is a protected location")
    assert stall < fence, "the stall branch must be taken before the protected-location wording"


def test_a_read_only_verdict_reopens_on_a_changed_config() -> None:
    """A read-only resolution is against a path the operator named and may correct.

    Both reopen gates key on the verdict pair, so a corrected `repo_path` re-resolves
    instead of serving the stranger's repository until the gateway restarts.
    """
    for source in (
        inspect.getsource(repository._invalid_resolution_is_stale),
        inspect.getsource(repository.ensure_main_repo_discovered),
    ):
        assert "(_REPO_INVALID_MSG or _REPO_READ_ONLY_MSG) and MAIN_REPO" in source


# --- a repository that would execute code on a read is refused ---------------


def test_executable_filter_drivers_refuse_the_adoption() -> None:
    """A filter driver is a COMMAND, and ``git status`` is what would run it.

    The env neutralizers pin the named execution vectors but cannot enumerate driver
    names, so a repo-local filter is the one that stays reachable on a read. Reading
    config executes nothing, which is what makes asking first the whole remedy.
    """
    body = inspect.getsource(repository.ensure_main_repo_discovered)
    assert "_configured_filter_commands(discovered)" in body
    assert "elif filters:" in body
    # The unread scope is its own branch: refusing a repository for a driver nobody
    # saw asserts a measurement that was never taken.
    assert "elif filters_unread:" in body
    assert repository._FILTER_COMMAND_SUFFIXES == (".clean", ".smudge", ".process")


def _filter_probe_calls(
    monkeypatch: pytest.MonkeyPatch, answers: dict[str, tuple[int, str]]
) -> list[list[str]]:
    """Run the probe against scripted git answers, returning the argv it used.

    ``answers`` maps a distinguishing argv token to the ``(returncode, stdout)`` git
    should answer with. Nothing executes: the point is which questions are asked.
    """
    seen: list[list[str]] = []

    def _run(argv, **_kwargs):
        seen.append(list(argv))
        for token, (rc, out) in answers.items():
            if token in argv:
                return SimpleNamespace(returncode=rc, stdout=out)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(runtime, "_trusted_bin", lambda _name: "git")
    monkeypatch.setattr(repository.subprocess, "run", _run)
    return seen


def test_filter_probe_follows_includes_on_every_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``include.path`` resolves during a read, so the probe must follow it too.

    For a SPECIFIC scope git defaults include-following OFF, so a driver reached
    through ``[include] path = other.cfg`` answers an empty list to a probe that
    omits ``--includes`` while still executing on the next content-touching read.
    """
    seen = _filter_probe_calls(monkeypatch, {"extensions.worktreeConfig": (0, "true\n")})
    drivers, unread = repository._configured_filter_commands("/somewhere/other-project")

    assert (drivers, unread) == ([], None)
    config_reads = [argv for argv in seen if "config" in argv]
    assert config_reads
    for argv in config_reads:
        assert "--includes" in argv


def test_filter_probe_asks_the_worktree_scope_only_when_it_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--worktree`` off is not a scope git reads, and asking anyway errors.

    Without ``extensions.worktreeConfig`` git ignores ``config.worktree`` entirely
    and refuses the query outright on any repository that has a linked worktree —
    the ordinary shape of what this app enumerates — so an unconditional ask turns
    a filter-free repository into a refusal.
    """
    off = _filter_probe_calls(monkeypatch, {"extensions.worktreeConfig": (1, "")})
    assert repository._configured_filter_commands("/repo") == ([], None)
    assert not [argv for argv in off if "--worktree" in argv]

    on = _filter_probe_calls(monkeypatch, {"extensions.worktreeConfig": (0, "true\n")})
    assert repository._configured_filter_commands("/repo") == ([], None)
    assert [argv for argv in on if "--worktree" in argv]


def test_filter_probe_reports_an_unread_scope_apart_from_a_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable scope is a measurement nobody took, not a configured driver."""
    _filter_probe_calls(
        monkeypatch,
        {
            "extensions.worktreeConfig": (0, "true\n"),
            "--get-regexp": (128, ""),
        },
    )
    drivers, unread = repository._configured_filter_commands("/repo")

    assert drivers == []
    assert unread and "could not be read" in unread


def test_filter_probe_reports_a_reason_when_git_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_trusted_bin", lambda _name: None)
    drivers, unread = repository._configured_filter_commands("/somewhere/other-project")
    assert drivers == []
    assert unread and "unverified" in unread


def test_every_read_on_a_read_only_checkout_re_takes_the_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery's answer is about the repository as it was THEN.

    A driver written into the config after the adoption is read by the next git
    invocation and by nothing else, so the clearance is re-taken at the chokepoint
    every read passes through, and the repository stops being served at once.
    """
    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", "read-only: /repo ...")
    monkeypatch.setattr(
        repository,
        "_configured_filter_commands",
        lambda _path: (["filter.evil.process"], None),
    )

    async def _never(*_args, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("git ran against a checkout that would execute its driver")

    monkeypatch.setattr(runtime, "_run_cmd", _never)

    with pytest.raises(repository.RepoUnreadable) as caught:
        asyncio.run(repository._git("/repo", "status", "--porcelain"))
    assert "filter.evil.process" in str(caught.value)


def test_the_product_s_own_checkout_pays_no_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trust boundary predates this mode: its config is the operator's own."""
    monkeypatch.setattr(repository, "_REPO_READ_ONLY_MSG", None)

    def _refuse(_path):  # pragma: no cover - must not be reached
        raise AssertionError("probed the checkout this app owns")

    monkeypatch.setattr(repository, "_configured_filter_commands", _refuse)

    async def _ok(*_args, **_kwargs):
        return 0, "clean\n", ""

    monkeypatch.setattr(runtime, "_run_cmd", _ok)
    assert asyncio.run(repository._git("/repo", "status", "--porcelain")) == "clean"


def test_the_poll_path_reopens_a_read_only_latch() -> None:
    """The reopen-on-changed-config path runs through the fleet poll, not only /fleet.

    ``_ensure_repo_resolved`` returns before ``ensure_main_repo_discovered`` for a
    resolved checkout, so a short-circuit that reads only the invalid verdict keeps
    serving a stranger's repository — base branch and upstream remote resolved
    against it — until the gateway restarts.
    """
    source = inspect.getsource(worktree_ops._ensure_repo_resolved)
    assert "not repository._REPO_INVALID_MSG" in source
    assert "not repository._REPO_READ_ONLY_MSG" in source


# --- a read-only checkout reports its live state as unknown -------------------


def test_read_only_checkout_reports_live_state_unknown() -> None:
    """`null` badges are falsy in the page, so the refusal needs the unknown flag.

    Make live is refused on a read-only checkout, and the page already disables the
    control and states the reason when the live state is unknown. Without this the
    row renders as "nothing is live" and still offers the button that answers 409.
    """
    body = inspect.getsource(fleet_state._build_fleet)
    assert "if repository._read_only_reason():" in body
    gate = body.index("if repository._read_only_reason():")
    assert body.index("live_state_known = False", gate) > gate


# --- the read-only denial reaches the audit trail ----------------------------


def test_read_only_denial_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal on an AUTHENTICATED request is a permission decision.

    Recorded on BOTH surfaces that take it: the backend middleware, whose HMAC
    denials beside it are already audited, and the gateway's own resolution step,
    which returns before the route reaches its ``_audit`` call.
    """
    for source in (
        inspect.getsource(http_api.hmac_proxy_middleware),
        inspect.getsource(gateway_routes._ensure_repo),
    ):
        assert "log_tool_invocation" in source
        assert 'outcome="denied"' in source
        assert 'tool_name="dev-fleet:repo-read-only"' in source
        # The 409 must survive a failing audit sink: auditing may not mask the answer.
        assert "except Exception" in source
