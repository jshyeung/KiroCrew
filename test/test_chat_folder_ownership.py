"""Ownership on the chat-folder tree-shaping endpoints.

A folder created by an app carries it in ``owner_app``; an absent key reads as
the person's, which is what makes this a field addition rather than a migration.
An app may create at the top level or inside a folder it owns, and may rename,
reparent or delete only what it owns. The person is never confined.

The scope is derived from the authenticated calling session, never the body: the
managed MCP set authenticates with the internal secret, which carries no app
claim, so an app agent's tool call arrives with ``request["app"]`` empty.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.dashboard.chat_folders import (
    _resolve_folder_project_dir,
    _validate_project_dir,
    api_chat_folder_create,
    api_chat_folder_delete,
    api_chat_folder_update,
)
from kiro_crew.dashboard.state import DashboardState, _ChatSlot

# fldr…01 belongs to the person, …02 to issue-radar, …03 to another app, and
# …04 predates the field entirely (no key at all) — the legacy row.
PERSON = "fldr00000001"
RADAR = "fldr00000002"
OTHER = "fldr00000003"
LEGACY = "fldr00000004"


def _folders() -> list[dict[str, Any]]:
    return [
        {"id": PERSON, "name": "Work", "parent_id": "", "owner_app": ""},
        {"id": RADAR, "name": "Radar output", "parent_id": "", "owner_app": "issue-radar"},
        {"id": OTHER, "name": "Specs", "parent_id": "", "owner_app": "spec-builder"},
        {"id": LEGACY, "name": "Old", "parent_id": ""},
    ]


def _app_slot(key: str, app: str) -> _ChatSlot:
    slot = _ChatSlot(key)
    slot._app = app
    return slot


def _state(*slots: _ChatSlot, folders: list[dict[str, Any]] | None = None) -> DashboardState:
    state = MagicMock(spec=DashboardState)
    state._folders = _folders() if folders is None else folders
    state._slots = {s.key: s for s in slots}
    state.push_slots_update = MagicMock()
    # No archive by default: _folder_history_counts returns {} early on a falsy
    # conversation_log, which is what an app's delete consults for emptiness. A
    # bare MagicMock here would be iterated instead and raise.
    state.conversation_log = None

    async def _mutate(fn: Any, on_committed: Any = None) -> Any:
        # The real store runs the callback under a lock and hands back its
        # second element; the ownership decisions live inside that callback, so a
        # mock that never calls it would prove nothing.
        changed, value = fn(state._folders)
        if changed and on_committed is not None:
            on_committed()
        return value

    state.mutate_folders = AsyncMock(side_effect=_mutate)
    return state


def _make_app(state: DashboardState) -> web.Application:
    app = web.Application()
    app["state"] = state

    @web.middleware
    async def _publish_app(request: web.Request, handler: Any) -> Any:
        # Stands in for the token middleware. Empty for the internal-secret
        # (MCP) transport, which is the path an app agent's tool call takes.
        request["app"] = ""
        return await handler(request)

    app.middlewares.append(_publish_app)
    app.router.add_post("/api/chat/folders", api_chat_folder_create)
    app.router.add_patch("/api/chat/folders/{id}", api_chat_folder_update)
    app.router.add_delete("/api/chat/folders/{id}", api_chat_folder_delete)
    return app


def _by_id(state: DashboardState, fid: str) -> dict[str, Any] | None:
    return next((f for f in state._folders if f["id"] == fid), None)


class TestOrderIsStoredVerbatim:
    """The endpoint stores whatever int the body carries, sign included.

    ``chat_folder_move``'s free-slot placement puts a folder ahead of the first
    sibling by writing ``first.order - 1``, which is NEGATIVE once the sidebar has
    renumbered a set from 0 — the ordinary case. Nothing in the tool layer can make
    that work if the endpoint clamps or rejects it, and the tool writes it as the
    single request that keeps a reposition from landing half-applied.
    """

    @pytest.mark.asyncio
    async def test_a_negative_order_is_accepted(self) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"order": -1},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, PERSON)["order"] == -1

    @pytest.mark.asyncio
    async def test_a_gap_midpoint_is_accepted(self) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"order": 5},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, PERSON)["order"] == 5

    @pytest.mark.asyncio
    async def test_a_duplicate_order_is_not_refused(self) -> None:
        """Two siblings may share a number; the name tie-break resolves them.

        The free-slot check treats equal neighbours as no room precisely because
        the store allows this, so the allowance has to be pinned.
        """
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            first = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"order": 7},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            second = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"order": 7},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert (first.status, second.status) == (200, 200)
        assert _by_id(state, PERSON)["order"] == 7
        assert _by_id(state, RADAR)["order"] == 7


class TestTheToolPreCheckMatchesTheEndpointRule:
    """The tool's renumber pre-check and this endpoint must agree on ownership.

    ``chat_folder_move`` refuses an app a placement that would renumber a row it
    does not own, and it decides that in the TOOL layer, before its first write —
    because the endpoint judges one row at a time, so a refusal arriving halfway
    leaves the sidebar in an order nobody chose. That means the same rule is
    expressed twice: ``owner_app``-vs-caller in ``mcp_dashboard`` and
    ``_folder_owner_app`` inside this endpoint's ``_apply``.

    These drive the real endpoint rather than a patched ``_patch``, so a change to
    either side's rule — a tightened check, a different absent-key default — turns
    one of them red instead of letting the pre-check quietly permit a write the
    endpoint then refuses (or refuse one it would have allowed).
    """

    @pytest.mark.asyncio
    async def test_the_endpoint_refuses_the_order_write_the_pre_check_refuses(self) -> None:
        """An app writing order on a foreign row: refused, exactly as pre-checked."""
        state = _state(_app_slot("chat-1-200", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"order": 3},
                headers={"X-Session-Key": "dashboard:chat-1-200"},
            )
        assert resp.status == 403
        assert "order" not in (_by_id(state, PERSON) or {})

    @pytest.mark.asyncio
    async def test_the_endpoint_allows_the_order_write_the_pre_check_allows(self) -> None:
        """The same app on its OWN row: allowed, so the pre-check is not over-broad."""
        state = _state(_app_slot("chat-1-200", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"order": 3},
                headers={"X-Session-Key": "dashboard:chat-1-200"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["order"] == 3

    @pytest.mark.asyncio
    async def test_a_legacy_row_reads_as_the_persons_on_both_sides(self) -> None:
        """The absent-key default is the drift the pre-check is most exposed to.

        ``LEGACY`` carries no ``owner_app`` at all. The pre-check reads a missing
        key as the person's via ``.get("owner_app")``; if the endpoint ever read it
        as unowned instead, an app renumber would sail past the pre-check and land.
        """
        state = _state(_app_slot("chat-1-200", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{LEGACY}",
                json={"order": 9},
                headers={"X-Session-Key": "dashboard:chat-1-200"},
            )
        assert resp.status == 403
        assert "order" not in (_by_id(state, LEGACY) or {})


class TestCreateStampsTheOwner:
    @pytest.mark.asyncio
    async def test_an_apps_folder_is_stamped_with_that_app(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201
        assert body["owner_app"] == "issue-radar"

    @pytest.mark.asyncio
    async def test_the_persons_folder_carries_no_owner_key(self) -> None:
        """Absent, not empty-string: "absent means the person" stays the one
        representation, and the person's rows keep the shape they have on disk."""
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Q3"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201
        assert "owner_app" not in body

    @pytest.mark.asyncio
    async def test_the_owner_is_never_taken_from_the_body(self) -> None:
        """A caller that could name its own owner could name someone else's."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "owner_app": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 201
        assert body["owner_app"] == "issue-radar"

    @pytest.mark.asyncio
    async def test_an_app_may_nest_under_its_own_folder(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "parent_id": RADAR},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 201

    @pytest.mark.asyncio
    async def test_an_app_may_not_nest_under_the_persons_folder(self) -> None:
        """Nesting writes to THAT folder's child list."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        before = len(state._folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "parent_id": PERSON},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_not_owned"
        assert len(state._folders) == before

    @pytest.mark.asyncio
    async def test_a_legacy_row_without_the_key_is_the_persons(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "parent_id": LEGACY},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 403


class TestRenameAndReparentAreBounded:
    @pytest.mark.asyncio
    async def test_an_app_can_rename_its_own_folder(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Renamed"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Renamed"

    @pytest.mark.asyncio
    async def test_an_app_cannot_rename_the_persons_folder(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"name": "Hijacked"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_not_owned"
        assert _by_id(state, PERSON)["name"] == "Work"

    @pytest.mark.asyncio
    async def test_an_app_cannot_rename_another_apps_folder(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{OTHER}",
                json={"name": "Hijacked"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 403
        assert _by_id(state, OTHER)["name"] == "Specs"

    @pytest.mark.asyncio
    async def test_the_person_is_not_confined_by_an_apps_ownership(self) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Tidied up"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Tidied up"

    @pytest.mark.asyncio
    async def test_an_app_cannot_reparent_its_folder_into_the_persons(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": PERSON},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_not_owned"
        assert _by_id(state, RADAR)["parent_id"] == ""

    @pytest.mark.asyncio
    async def test_an_app_can_reparent_to_the_top_level(self) -> None:
        """The top level is not a folder row, so it has no owner to violate —
        that is where an app's own tree starts."""
        folders = _folders()
        nested = {
            "id": "fldr00000005",
            "name": "Runs",
            "parent_id": RADAR,
            "owner_app": "issue-radar",
        }
        folders.append(nested)
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                "/api/chat/folders/fldr00000005",
                json={"parent_id": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, "fldr00000005")["parent_id"] == ""

    @pytest.mark.asyncio
    async def test_ownership_cannot_be_reassigned_by_a_patch(self) -> None:
        """Stamped once at create; not a field a request can hand over or clear."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"owner_app": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["owner_app"] == "issue-radar"

    @pytest.mark.asyncio
    async def test_moving_own_folder_that_holds_a_foreign_one_is_refused(self) -> None:
        """A move takes the subtree with it, so the person's nested folder would
        be relocated by an app's write."""
        folders = _folders()
        folders.append({"id": "fldr00000007", "name": "Theirs", "parent_id": RADAR})
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_not_owned"
        assert _by_id(state, RADAR)["parent_id"] == ""

    @pytest.mark.asyncio
    async def test_renaming_a_folder_that_holds_a_foreign_one_is_still_allowed(self) -> None:
        """Only the MOVE is gated on the subtree -- a rename relocates nothing."""
        folders = _folders()
        folders.append({"id": "fldr00000007", "name": "Theirs", "parent_id": RADAR})
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Renamed"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Renamed"

    @pytest.mark.asyncio
    async def test_the_person_can_still_move_a_folder_holding_an_apps(self) -> None:
        """Containment cuts both ways, but the person is never confined."""
        folders = _folders()
        folders.append(
            {
                "id": "fldr00000007",
                "name": "Radar sub",
                "parent_id": PERSON,
                "owner_app": "issue-radar",
            }
        )
        state = _state(_ChatSlot("chat-1-100"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"parent_id": OTHER},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, PERSON)["parent_id"] == OTHER


class TestAnAppCannotChangeAnExistingFoldersProjectDir:
    """A folder's binding is what every session filed in its subtree picks up
    on its next agent switch, and those sessions live in stores the folder store
    shares no lock with (the slot table, the session archive). "Every session
    under this folder is the caller's own" cannot be established atomically with
    the write, so the PATCH refuses an agent principal's ``project_dir`` change
    outright -- the delete route's rule, on the binding axis. An agent principal
    binds a folder at CREATE, when nothing is filed in it yet; the person keeps
    the update they always had.
    """

    @staticmethod
    def _radar_holding_theirs(bound: str = "") -> list[dict[str, Any]]:
        folders = _folders()
        if bound:
            next(f for f in folders if f["id"] == RADAR)["project_dir"] = bound
        folders.append({"id": "fldr00000007", "name": "Theirs", "parent_id": RADAR})
        return folders

    @pytest.mark.asyncio
    async def test_an_app_cannot_rebind_its_folder_holding_the_persons_chat(self, tmp_path) -> None:
        """The person filed one of their own chats into the app's folder. That
        slot re-resolves the folder's binding on its next agent switch
        (``api_chat_slot_agent``), so an app rebinding the folder would decide
        the person's project, cwd and steering -- cross-ownership through a
        session the folder store cannot see atomically."""
        theirs = _ChatSlot("chat-2-200")
        theirs.folder_id = RADAR
        state = _state(_app_slot("chat-1-100", "issue-radar"), theirs)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert "project_dir" not in _by_id(state, RADAR)

    @pytest.mark.asyncio
    async def test_binding_own_folder_that_holds_a_foreign_one_is_refused(self, tmp_path) -> None:
        """The nested-folder case is one instance of the same rule: a chat the
        person opens in the nested folder would inherit the app's binding."""
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=self._radar_holding_theirs())
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert "project_dir" not in _by_id(state, RADAR)

    @pytest.mark.asyncio
    async def test_clearing_is_refused_the_same_way(self, tmp_path) -> None:
        """Clearing changes what a filed session picks up just as setting does."""
        state = _state(
            _app_slot("chat-1-100", "issue-radar"),
            folders=self._radar_holding_theirs(bound=str(tmp_path)),
        )
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"project_dir": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 403
        assert _by_id(state, RADAR)["project_dir"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_even_a_subtree_of_only_its_own_folders_and_sessions_is_refused(
        self, tmp_path
    ) -> None:
        """No narrower rule: an own-folders-only subtree with only the app's own
        live session filed in it is refused too, because the archive (no owner
        in its index, sessions revive with ``folder_id`` intact) and a filing
        that lands mid-request are exactly what the folder store cannot see.
        Nothing is stored and the path is never validated."""
        folders = _folders()
        folders.append(
            {"id": "fldr00000007", "name": "Runs", "parent_id": RADAR, "owner_app": "issue-radar"}
        )
        own = _app_slot("chat-1-100", "issue-radar")
        own.folder_id = RADAR
        state = _state(own, folders=folders)
        with patch("kiro_crew.dashboard.chat_folders._validate_project_dir") as validator:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{RADAR}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
                body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert "bind it when creating the folder" in body["error"]
        assert "project_dir" not in _by_id(state, RADAR)
        validator.assert_not_called()
        state.mutate_folders.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited(self, tmp_path) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{RADAR}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == "issue-radar"
        assert kwargs["operation"] == "chat.folder_update"
        assert kwargs["outcome"] == "denied"
        assert kwargs["resources"] == RADAR
        assert "project directory" in kwargs["error"]

    @pytest.mark.asyncio
    async def test_an_apps_other_fields_on_its_own_folder_still_apply(self, tmp_path) -> None:
        """The rule is about the binding only: a rename of the same folder by
        the same app lands as before."""
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Radar runs"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Radar runs"

    @pytest.mark.asyncio
    async def test_the_person_can_bind_a_folder_holding_an_apps(self, tmp_path) -> None:
        folders = _folders()
        folders.append(
            {
                "id": "fldr00000007",
                "name": "Radar sub",
                "parent_id": PERSON,
                "owner_app": "issue-radar",
            }
        )
        state = _state(_ChatSlot("chat-1-100"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, PERSON)["project_dir"] == str(tmp_path.resolve())

    @pytest.mark.asyncio
    async def test_the_path_validator_runs_off_the_event_loop(self, tmp_path) -> None:
        """realpath/isdir on a stalled network path must not hold the gateway
        loop: the PATCH route hands the validator to a worker thread, as the
        create route does."""
        state = _state(_ChatSlot("chat-1-100"))
        seen: list[Any] = []
        real_to_thread = asyncio.to_thread

        async def _spy(fn: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(fn)
            return await real_to_thread(fn, *args, **kwargs)

        with patch("kiro_crew.dashboard.chat_folders.asyncio.to_thread", side_effect=_spy):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{PERSON}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 200
        assert _validate_project_dir in seen


class TestAChannelAgentCannotBindAFolder:
    """A Channels agent (session key ``channel:<channel_id>:<agent_id>``) acts on
    words from a thread other people are in. Its key names no dashboard slot and
    no app, so ``folder_principal`` reads it as the PERSON -- and the app/member
    fence on the two binding paths is keyed on that principal. Without a fence
    of its own, a channel agent could bind a new folder or rebind an existing
    one with the person's full authority. Both paths refuse it with the same 403
    the agent-principal fence answers; the person's authority is untouched, and
    a channel agent's other folder writes are not this rule's concern.
    """

    CHANNEL = "channel:chan-000001:helper"

    @pytest.mark.asyncio
    async def test_a_channel_agent_cannot_create_a_bound_folder(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        before = len(state._folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": self.CHANNEL},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert len(state._folders) == before

    @pytest.mark.asyncio
    async def test_a_channel_agent_cannot_change_an_existing_folders_binding(
        self, tmp_path
    ) -> None:
        """Set AND clear: the person's own folder, which a channel key would
        otherwise reach as the person."""
        folders = _folders()
        bound = {"id": "fldr00000008", "name": "Bound", "parent_id": "", "project_dir": "/t"}
        folders.append(bound)
        state = _state(_ChatSlot("chat-1-100"), folders=folders)
        with patch(
            "kiro_crew.dashboard.chat_folders._validate_project_dir",
            return_value=(str(tmp_path), None),
        ) as validator:
            async with TestClient(TestServer(_make_app(state))) as client:
                setting = await client.patch(
                    f"/api/chat/folders/{PERSON}",
                    json={"project_dir": str(tmp_path)},
                    headers={"X-Session-Key": self.CHANNEL},
                )
                set_body = await setting.json()
                clearing = await client.patch(
                    "/api/chat/folders/fldr00000008",
                    json={"project_dir": ""},
                    headers={"X-Session-Key": self.CHANNEL},
                )
        assert (setting.status, clearing.status) == (403, 403)
        assert set_body["code"] == "folder_project_dir_forbidden"
        assert "project_dir" not in _by_id(state, PERSON)
        assert _by_id(state, "fldr00000008")["project_dir"] == "/t"
        # Refused before the path is looked at, like the agent-principal fence.
        validator.assert_not_called()
        state.mutate_folders.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited_against_the_channel_key(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/folders",
                    json={"name": "Runs", "project_dir": str(tmp_path)},
                    headers={"X-Session-Key": self.CHANNEL},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == self.CHANNEL
        assert kwargs["operation"] == "chat.folder_create"
        assert kwargs["outcome"] == "denied"
        assert kwargs["source"] == "channel"
        assert "project directory" in kwargs["error"]

    @pytest.mark.asyncio
    async def test_a_channel_agents_unbound_folder_writes_are_not_this_rule(self, tmp_path) -> None:
        """Scope pin: the fence is on the BINDING. An unbound create and a
        rename by the same channel key land exactly as they did."""
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            created = await client.post(
                "/api/chat/folders",
                json={"name": "Runs"},
                headers={"X-Session-Key": self.CHANNEL},
            )
            renamed = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"name": "Work items"},
                headers={"X-Session-Key": self.CHANNEL},
            )
        assert (created.status, renamed.status) == (201, 200)
        assert _by_id(state, PERSON)["name"] == "Work items"

    @pytest.mark.asyncio
    async def test_the_person_still_binds_at_create_and_update(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"))
        async with TestClient(TestServer(_make_app(state))) as client:
            created = await client.post(
                "/api/chat/folders",
                json={"name": "Runs", "project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            updated = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"project_dir": str(tmp_path)},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert (created.status, updated.status) == (201, 200)
        assert _by_id(state, PERSON)["project_dir"] == str(tmp_path.resolve())


class TestAMoveCannotChangeWhatASubtreeInherits:
    """The composition the create-only rule leaves open: an agent principal
    binds a NEW folder at create (allowed -- nothing is filed in it yet), then
    reparents an EXISTING folder it owns under it. Every session filed in the
    moved subtree -- including one of the person's the folder store cannot see
    -- resolves the destination's binding on its next agent switch, so the move
    rebinds them exactly as the refused PATCH would have. The same holds in the
    other direction (moving out from under a binding clears it).

    Rule: an agent principal's reparent may not change what the moved subtree
    inherits. A folder with a binding of its own moves freely (its subtree
    resolves to it first); an unbound one may move only between places whose
    inherited binding is the same. The person is never confined.
    """

    BOUND = "fldr00000009"
    BOUND_CHILD = "fldr00000010"

    @staticmethod
    def _tree(bound_dir: str, radar_parent: str = "") -> list[dict[str, Any]]:
        folders = _folders()
        next(f for f in folders if f["id"] == RADAR)["parent_id"] = radar_parent
        folders.append(
            {
                "id": TestAMoveCannotChangeWhatASubtreeInherits.BOUND,
                "name": "Bound at create",
                "parent_id": "",
                "owner_app": "issue-radar",
                "project_dir": bound_dir,
            }
        )
        folders.append(
            {
                "id": TestAMoveCannotChangeWhatASubtreeInherits.BOUND_CHILD,
                "name": "Under bound",
                "parent_id": TestAMoveCannotChangeWhatASubtreeInherits.BOUND,
                "owner_app": "issue-radar",
            }
        )
        return folders

    @pytest.mark.asyncio
    async def test_an_app_cannot_move_its_folder_under_one_it_bound_at_create(
        self, tmp_path
    ) -> None:
        """The exact composition: create C with project_dir, then reparent the
        app's existing folder -- holding one of the person's chats -- under C.
        Before: the move lands and the person's chat resolves C's directory.
        After: refused with the binding-fence code, nothing inherited."""
        theirs = _ChatSlot("chat-2-200")
        theirs.folder_id = RADAR
        state = _state(
            _app_slot("chat-1-100", "issue-radar"), theirs, folders=self._tree(str(tmp_path))
        )
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": self.BOUND},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert _by_id(state, RADAR)["parent_id"] == ""
        # What the person's chat resolves on its next agent switch: still nothing.
        assert _resolve_folder_project_dir(state._folders, RADAR) == ("", None)

    @pytest.mark.asyncio
    async def test_moving_out_from_under_a_binding_is_refused_the_same_way(self, tmp_path) -> None:
        """The clear direction: the subtree would stop inheriting."""
        state = _state(
            _app_slot("chat-1-100", "issue-radar"),
            folders=self._tree(str(tmp_path), radar_parent=self.BOUND),
        )
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": ""},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_project_dir_forbidden"
        assert _by_id(state, RADAR)["parent_id"] == self.BOUND

    @pytest.mark.asyncio
    async def test_a_move_that_keeps_the_inherited_binding_still_lands(self, tmp_path) -> None:
        """Between two places under the same bound ancestor nothing changes for
        the subtree, so the app's own tree stays organisable."""
        state = _state(
            _app_slot("chat-1-100", "issue-radar"),
            folders=self._tree(str(tmp_path), radar_parent=self.BOUND),
        )
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": self.BOUND_CHILD},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["parent_id"] == self.BOUND_CHILD

    @pytest.mark.asyncio
    async def test_a_folder_with_its_own_binding_moves_freely(self, tmp_path) -> None:
        """Nearest binding wins in the resolver, so a bound folder's subtree
        resolves to it wherever it sits -- the move changes nothing inherited."""
        folders = self._tree(str(tmp_path))
        next(f for f in folders if f["id"] == RADAR)["project_dir"] = str(tmp_path / "own")
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=folders)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": self.BOUND},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["parent_id"] == self.BOUND

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited(self, tmp_path) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"), folders=self._tree(str(tmp_path)))
        with patch("kiro_crew.dashboard.chat_folders.sel") as sel_fn:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.patch(
                    f"/api/chat/folders/{RADAR}",
                    json={"parent_id": self.BOUND},
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        kwargs = sel_fn.return_value.log_api_access.call_args.kwargs
        assert kwargs["caller"] == "issue-radar"
        assert kwargs["operation"] == "chat.folder_update"
        assert kwargs["outcome"] == "denied"
        assert kwargs["resources"] == RADAR
        assert "inherit" in kwargs["error"]

    @pytest.mark.asyncio
    async def test_the_person_moves_across_bindings_freely(self, tmp_path) -> None:
        state = _state(_ChatSlot("chat-1-100"), folders=self._tree(str(tmp_path)))
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"parent_id": self.BOUND},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["parent_id"] == self.BOUND


class TestAnAppCannotDeleteFolders:
    """A delete relocates everything the folder contains, and those contents live
    in a DIFFERENT store from the folder -- the slot table and the session
    archive, neither sharing a lock with it. So emptiness cannot be established
    atomically with the removal, and every narrower rule leaked through another
    seam. The person keeps the delete they always had.

    Nothing shipped loses a capability: no MCP tool exposes folder deletion, and
    the only client of the route is the dashboard UI.
    """

    @pytest.mark.asyncio
    async def test_an_app_cannot_delete_even_an_empty_folder_it_owns(self) -> None:
        state = _state(_app_slot("chat-1-100", "issue-radar"))
        with patch("kiro_crew.dashboard.chat_folders.save_slot_off_loop", AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.delete(
                    f"/api/chat/folders/{RADAR}",
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
                body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "folder_delete_forbidden"
        assert _by_id(state, RADAR) is not None

    @pytest.mark.asyncio
    async def test_no_session_is_touched_by_the_refusal(self) -> None:
        """Refused before the unfile loop, so nothing is written and there is
        nothing to roll back."""
        mine = _app_slot("chat-9-900", "issue-radar")
        mine.folder_id = RADAR
        state = _state(_app_slot("chat-1-100", "issue-radar"), mine)
        with patch("kiro_crew.dashboard.chat_folders.save_slot_off_loop", AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.delete(
                    f"/api/chat/folders/{RADAR}",
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        assert mine.folder_id == RADAR

    @pytest.mark.asyncio
    async def test_the_person_can_still_delete_a_full_folder(self) -> None:
        """The person is not confined: clearing out a folder full of
        conversations and subfolders is the delete they already had."""
        theirs = _app_slot("chat-9-900", "issue-radar")
        theirs.folder_id = RADAR
        folders = _folders()
        folders.append({"id": "fldr00000006", "name": "Sub", "parent_id": RADAR})
        state = _state(_ChatSlot("chat-1-100"), theirs, folders=folders)
        with patch("kiro_crew.dashboard.chat_folders.save_slot_off_loop", AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.delete(
                    f"/api/chat/folders/{RADAR}",
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 200
        assert _by_id(state, RADAR) is None
        assert theirs.folder_id == ""
        assert _by_id(state, "fldr00000006")["parent_id"] == ""


class TestACallerWhoseSlotIsGoneIsRefused:
    """An empty scope reads as the person, which is right for a caller that never
    had a slot (Slack, a channel session, the person's cron) and wrong for a
    `dashboard:` key, which NAMES one. A tab closing while its tool call is in
    flight pops the slot without draining, so an app-owned session would arrive
    unattributable and be handed the person's authority over the person's folders.
    """

    @pytest.mark.asyncio
    async def test_create_is_refused(self) -> None:
        state = _state()  # the named slot is absent from the registry
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Sneaky"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
            body = await resp.json()
        assert resp.status == 403
        assert body["code"] == "caller_unattributable"

    @pytest.mark.asyncio
    async def test_rename_of_the_persons_folder_is_refused(self) -> None:
        state = _state()
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{PERSON}",
                json={"name": "Hijacked"},
                headers={"X-Session-Key": "dashboard:chat-1-100"},
            )
        assert resp.status == 403
        assert _by_id(state, PERSON)["name"] == "Work"

    @pytest.mark.asyncio
    async def test_delete_is_refused_before_any_slot_is_unfiled(self) -> None:
        filed = _ChatSlot("chat-9-900")
        filed.folder_id = PERSON
        state = _state(filed)
        with patch("kiro_crew.dashboard.chat_folders.save_slot_off_loop", AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.delete(
                    f"/api/chat/folders/{PERSON}",
                    headers={"X-Session-Key": "dashboard:chat-1-100"},
                )
        assert resp.status == 403
        assert _by_id(state, PERSON) is not None
        assert filed.folder_id == PERSON

    @pytest.mark.asyncio
    async def test_a_caller_that_never_had_a_slot_is_still_the_person(self) -> None:
        """The refusal must not swallow Slack, channel or cron callers -- they
        never had a slot to be confined to, which is a different fact."""
        state = _state()
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.patch(
                f"/api/chat/folders/{RADAR}",
                json={"name": "Tidied"},
                headers={"X-Session-Key": "slack:T1/C1"},
            )
        assert resp.status == 200
        assert _by_id(state, RADAR)["name"] == "Tidied"
