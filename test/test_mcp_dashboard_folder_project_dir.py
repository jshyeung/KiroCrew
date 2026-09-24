"""``project_dir`` on the folder tools, driven through the REAL folder routes.

``test_mcp_dashboard_folders.py`` patches the HTTP helpers and pins the tool's
call shape. This module closes the other half of the tool's claim — that the
tool surfaces the endpoint's validation rather than a copy of it — by wiring
``mcp_dashboard``'s ``_get`` / ``_post`` / ``_patch`` to a live aiohttp test
server running ``chat_folders``' own handlers over a real ``DashboardState``,
then reading the store the routes wrote. The slot-create route is included so
the inheritance the feature exists for is observed end to end: a folder the
tool bound is one a session created inside it inherits from.

The tool is synchronous (it is a stdio MCP server), so it runs on a worker
thread while the bridge hands each request back to the test's event loop with
``run_coroutine_threadsafe`` — the same request/response contract
``mcp_core._send`` presents (a 2xx body verbatim; a 4xx collapsed to
``{"error", "code"}``).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_app_with_agent_routes, _make_folder_app, _make_state

from kiro_crew.dashboard.chat_folders import _resolve_folder_project_dir
from kiro_crew.mcp_dashboard import _call_tool_inner

CALLER = "chat-1-100"


class _Bridge:
    """``mcp_core`` request helpers re-targeted at an aiohttp ``TestClient``."""

    def __init__(self, client: TestClient, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop

    async def _request(
        self, method: str, path: str, body: dict | None, session_key: str | None
    ) -> Any:
        headers = {"X-Session-Key": session_key} if session_key else {}
        resp = await self._client.request(method, path, json=body, headers=headers)
        payload = await resp.json()
        if resp.status >= 400:
            # ``_http_error_body``'s flattening: the structured error, plus code.
            return {"error": str(payload.get("error")), "code": str(payload.get("code") or "")}
        return payload

    def _run(self, method: str, path: str, body: dict | None, session_key: str | None) -> Any:
        return asyncio.run_coroutine_threadsafe(
            self._request(method, path, body, session_key), self._loop
        ).result(timeout=30)

    # Signatures mirror ``mcp_core._get`` / ``_post`` / ``_patch``.
    def get(self, path: str, session_key: str | None = None, *, timeout: float = 10) -> Any:
        return self._run("GET", path, None, session_key)

    def post(
        self,
        path: str,
        body: dict | None = None,
        *,
        timeout: float = 30,
        session_key: str | None = None,
    ) -> dict:
        return self._run("POST", path, body or {}, session_key)

    def patch(self, path: str, body: dict | None = None, *, session_key: str | None = None) -> dict:
        return self._run("PATCH", path, body or {}, session_key)


async def _call(
    bridge: _Bridge, name: str, args: dict[str, Any], *, caller_key: str = f"dashboard:{CALLER}"
) -> str:
    """Run one tool call on a worker thread against the live routes.

    ``caller_key`` is the verified key the strict resolver hands the tool -- the
    dashboard slot by default, or another namespace (a ``channel:`` agent) to
    drive the routes as that principal.
    """
    with (
        patch("kiro_crew.mcp_dashboard._get", side_effect=bridge.get),
        patch("kiro_crew.mcp_dashboard._post", side_effect=bridge.post),
        patch("kiro_crew.mcp_dashboard._patch", side_effect=bridge.patch),
        patch(
            "kiro_crew.mcp_core._resolve_session_key_strict",
            return_value=caller_key,
        ),
    ):
        return await asyncio.to_thread(_call_tool_inner, name, args)


def _folder(state: Any, fid: str) -> dict[str, Any]:
    return next(f for f in state._folders if f["id"] == fid)


def _created_id(out: str) -> str:
    assert "(id=" in out, out
    return out.split("(id=", 1)[1].split(")", 1)[0]


@pytest.fixture
def state(tmp_path: Any, monkeypatch: Any) -> Any:
    monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
    st = _make_state(tmp_path)
    # The caller's own live slot: the tree-shaping gate scopes the caller by
    # finding this row, and the routes refuse a ``dashboard:`` key naming a slot
    # that is gone. Created with no app, so the caller is the person.
    st.get_or_create_slot(CALLER)
    return st


class TestCreateThroughTheRealRoute:
    @pytest.mark.asyncio
    async def test_a_valid_project_dir_is_stored_and_reported_canonically(
        self, state: Any, tmp_path: Any
    ) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        link = tmp_path / "proj-link"
        link.symlink_to(proj, target_is_directory=True)
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Proj", "project_dir": str(link)}
            )
        assert out.startswith("Created folder `Proj`"), out
        stored = _folder(state, _created_id(out))
        # The endpoint's validator canonicalised the path; the store and the
        # tool's report agree on the resolved form, which is what a session
        # created here will inherit.
        assert stored["project_dir"] == os.path.realpath(str(proj))
        assert f"Project directory: {os.path.realpath(str(proj))}" in out

    @pytest.mark.asyncio
    async def test_a_session_created_in_the_folder_inherits_the_binding(
        self, state: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """The point of the feature, observed: create the folder with the tool,
        then open a chat in it the way the dashboard does, and the slot starts
        with the folder's project."""
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Proj", "project_dir": str(proj)}
            )
        fid = _created_id(out)

        # Same pins the route's own inheritance test uses (test_dashboard_chat):
        # no configured default project, no eager spawn, a string default agent.
        mock_cfg = MagicMock()
        mock_cfg.dashboard.default_project = ""
        mock_cfg.default_agent = ""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.default_project_dir",
            lambda _workspace: str(tmp_path / "workspace-default"),
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.schedule_eager_spawn",
            lambda *_args, **_kwargs: None,
        )
        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots", json={"name": "in-proj", "folder_id": fid})
            data = await resp.json()
        assert resp.status == 200, data
        assert data["folder_id"] == fid
        assert data["project"] == os.path.realpath(str(proj))
        assert state._slots["in-proj"].project == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_a_sensitive_path_is_refused_with_the_routes_text_and_no_folder(
        self, state: Any
    ) -> None:
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Keys", "project_dir": "~/.ssh"}
            )
        assert out == "Error: project_dir refers to a sensitive path"
        assert state._folders == []

    @pytest.mark.asyncio
    async def test_a_missing_directory_is_refused_with_the_routes_text_and_no_folder(
        self, state: Any, tmp_path: Any
    ) -> None:
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge,
                "chat_folder_create",
                {"name": "Ghost", "project_dir": str(tmp_path / "does-not-exist")},
            )
        assert out == "Error: Project directory must be an existing directory"
        assert state._folders == []

    @pytest.mark.asyncio
    async def test_a_relative_path_is_refused_with_the_routes_text_and_no_folder(
        self, state: Any
    ) -> None:
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Rel", "project_dir": "projects/rel"}
            )
        assert out == "Error: Project directory must be an absolute path"
        assert state._folders == []

    @pytest.mark.asyncio
    async def test_a_create_without_project_dir_stores_an_empty_binding(self, state: Any) -> None:
        """Pin: a call without ``project_dir`` is byte-for-byte the same request, so the
        store row it produces is the one it always produced."""
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(bridge, "chat_folder_create", {"name": "Plain"})
        stored = _folder(state, _created_id(out))
        assert stored["project_dir"] == ""
        assert out == f"Created folder `Plain` (id={stored['id']})."


class TestUpdateThroughTheRealRoute:
    @pytest.mark.asyncio
    async def test_sets_then_clears(self, state: Any, tmp_path: Any) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(bridge, "chat_folder_create", {"name": "Work"})
            fid = _created_id(made)
            assert _folder(state, fid)["project_dir"] == ""

            out = await _call(
                bridge, "chat_folder_update", {"folder": "Work", "project_dir": str(proj)}
            )
            assert out.startswith("Set the project directory of `Work`"), out
            assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))

            out = await _call(bridge, "chat_folder_update", {"folder": fid, "project_dir": ""})
            assert out.startswith("Cleared the project directory of `Work`"), out
            assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_a_null_project_dir_clears_through_the_real_route(
        self, state: Any, tmp_path: Any
    ) -> None:
        """``null`` reaches the endpoint as the ``""`` clear, never as the string
        ``"None"`` -- which the route would refuse as a relative path, leaving
        the old binding in place behind an error."""
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(
                bridge, "chat_folder_create", {"name": "Work", "project_dir": str(proj)}
            )
            fid = _created_id(made)
            assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))

            out = await _call(bridge, "chat_folder_update", {"folder": fid, "project_dir": None})
            assert out.startswith("Cleared the project directory of `Work`"), out
            assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_an_invalid_path_changes_nothing(self, state: Any, tmp_path: Any) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(
                bridge, "chat_folder_create", {"name": "Work", "project_dir": str(proj)}
            )
            fid = _created_id(made)
            for bad, text in (
                ("~/.aws", "project_dir refers to a sensitive path"),
                (str(tmp_path / "gone"), "Project directory must be an existing directory"),
            ):
                out = await _call(
                    bridge, "chat_folder_update", {"folder": "Work", "project_dir": bad}
                )
                assert out == f"Error: {text}"
                assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_an_app_cannot_bind_the_persons_folder(self, state: Any, tmp_path: Any) -> None:
        """Ownership is the endpoint's, under the store lock: the app's PATCH
        reaches it under the app's verified key and comes back refused."""
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(bridge, "chat_folder_create", {"name": "Mine"})
            fid = _created_id(made)
            assert _folder(state, fid).get("owner_app", "") == ""

            # The same caller slot, now owned by an app (what an app-created
            # session's row carries), so the endpoint derives that app.
            state._slots[CALLER]._app = "issue-radar"
            out = await _call(
                bridge, "chat_folder_update", {"folder": "Mine", "project_dir": str(proj)}
            )
        assert out.startswith(
            "Error: an app or crew member cannot change an existing folder's project directory"
        ), out
        assert "chat_folder_create with project_dir" in out
        assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_an_app_binds_at_create_but_cannot_change_its_own_folder_afterwards(
        self, state: Any, tmp_path: Any
    ) -> None:
        """The capability an agent principal keeps is the CREATE-time binding;
        an existing folder's binding -- even its own, holding nothing but its
        own work -- is the person's to change."""
        proj = tmp_path / "proj"
        proj.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        state._slots[CALLER]._app = "issue-radar"
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(
                bridge, "chat_folder_create", {"name": "Radar output", "project_dir": str(proj)}
            )
            fid = _created_id(made)
            assert _folder(state, fid)["owner_app"] == "issue-radar"
            assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))
            for value in (str(other), ""):
                out = await _call(
                    bridge, "chat_folder_update", {"folder": "Radar output", "project_dir": value}
                )
                assert out.startswith(
                    "Error: an app or crew member cannot change an existing folder's "
                    "project directory"
                ), out
                assert _folder(state, fid)["project_dir"] == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_an_app_cannot_rebind_its_folder_holding_the_persons_chat(
        self, state: Any, tmp_path: Any
    ) -> None:
        """The case the rule exists for: the person filed one of their own chats
        into the app's folder. That slot picks the folder's binding up on its
        next agent switch, so the app's rebind would decide the person's
        project -- refused through the real route, binding unchanged."""
        proj = tmp_path / "proj"
        proj.mkdir()
        state._slots[CALLER]._app = "issue-radar"
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(bridge, "chat_folder_create", {"name": "Radar output"})
            fid = _created_id(made)
            theirs = state.get_or_create_slot("chat-2-200")
            theirs.folder_id = fid
            out = await _call(
                bridge,
                "chat_folder_update",
                {"folder": "Radar output", "project_dir": str(proj)},
            )
        assert out.startswith(
            "Error: an app or crew member cannot change an existing folder's project directory"
        ), out
        assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_an_app_cannot_bind_its_folder_once_the_person_nested_one_inside(
        self, state: Any, tmp_path: Any
    ) -> None:
        """One instance of the same rule: a chat the person opens in the folder
        they nested inside the app's would inherit the app's binding."""
        proj = tmp_path / "proj"
        proj.mkdir()
        state._slots[CALLER]._app = "issue-radar"
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            made = await _call(bridge, "chat_folder_create", {"name": "Radar output"})
            fid = _created_id(made)
            assert _folder(state, fid)["owner_app"] == "issue-radar"
            # The person nests a folder of their own inside the app's.
            state._folders.append(
                {"id": "fldr0000theirs", "name": "Theirs", "parent_id": fid, "owner_app": ""}
            )
            out = await _call(
                bridge,
                "chat_folder_update",
                {"folder": "Radar output", "project_dir": str(proj)},
            )
        assert out.startswith(
            "Error: an app or crew member cannot change an existing folder's project directory"
        ), out
        assert _folder(state, fid)["project_dir"] == ""


class TestAChannelAgentThroughTheRealRoute:
    """A Channels agent's key (``channel:<channel_id>:<agent_id>``) names no slot
    and no app, so the tree-shaping gate scopes it as the person and the routes
    derive no principal for it. The binding fence is the endpoint's, on both
    paths, keyed on the channel key itself -- observed here through the tool."""

    CHANNEL = "channel:chan-000001:helper"

    @pytest.mark.asyncio
    async def test_a_channel_agent_cannot_bind_at_create_or_update(
        self, state: Any, tmp_path: Any
    ) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge,
                "chat_folder_create",
                {"name": "Bound", "project_dir": str(proj)},
                caller_key=self.CHANNEL,
            )
            assert out == (
                "Error: a channel agent cannot set or clear a folder's project directory - "
                "ask the person"
            ), out
            assert not any(f["name"] == "Bound" for f in state._folders)

            # The person creates an unbound folder; the channel agent may not
            # bind it, and is NOT pointed at a create-time binding it is refused
            # just the same.
            made = await _call(bridge, "chat_folder_create", {"name": "Work"})
            fid = _created_id(made)
            out = await _call(
                bridge,
                "chat_folder_update",
                {"folder": "Work", "project_dir": str(proj)},
                caller_key=self.CHANNEL,
            )
            assert out == (
                "Error: a channel agent cannot set or clear a folder's project directory - "
                "ask the person"
            ), out
            assert "chat_folder_create" not in out
            assert _folder(state, fid)["project_dir"] == ""

    @pytest.mark.asyncio
    async def test_a_channel_agents_unbound_create_still_lands(
        self, state: Any, tmp_path: Any
    ) -> None:
        """Scope pin: the fence is on the binding, not on the channel agent's
        other folder writes."""
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            out = await _call(
                bridge, "chat_folder_create", {"name": "Notes"}, caller_key=self.CHANNEL
            )
        assert out.startswith("Created folder `Notes`"), out


class TestAMoveCannotRouteABindingOntoThePersonsChat:
    """The composition the create-only rule leaves open, through the real tools:
    the app binds a NEW folder at create (allowed), then moves its EXISTING
    folder -- holding one of the person's chats -- under it. The chat would
    resolve the new folder's directory on its next agent switch."""

    @pytest.mark.asyncio
    async def test_the_move_is_refused_and_nothing_is_inherited(
        self, state: Any, tmp_path: Any
    ) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        state._slots[CALLER]._app = "issue-radar"
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            bound = _created_id(
                await _call(
                    bridge, "chat_folder_create", {"name": "Bound", "project_dir": str(proj)}
                )
            )
            radar = _created_id(await _call(bridge, "chat_folder_create", {"name": "Radar output"}))
            theirs = state.get_or_create_slot("chat-2-200")
            theirs.folder_id = radar

            out = await _call(
                bridge, "chat_folder_move", {"folder": "Radar output", "new_parent": "Bound"}
            )
        assert out.startswith(
            "Error: an app or crew member cannot move a folder where its sessions would "
            "inherit a different project directory"
        ), out
        assert _folder(state, radar)["parent_id"] == ""
        assert _resolve_folder_project_dir(state._folders, radar) == ("", None)
        assert _folder(state, bound)["project_dir"] == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_a_folder_bound_at_create_still_moves_under_another(
        self, state: Any, tmp_path: Any
    ) -> None:
        """Nearest binding wins, so a bound folder's subtree resolves to it
        wherever it sits: the app keeps organising its own bound folders."""
        proj = tmp_path / "proj"
        proj.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        state._slots[CALLER]._app = "issue-radar"
        async with TestClient(TestServer(_make_folder_app(state))) as client:
            bridge = _Bridge(client, asyncio.get_running_loop())
            bound = _created_id(
                await _call(
                    bridge, "chat_folder_create", {"name": "Bound", "project_dir": str(proj)}
                )
            )
            runs = _created_id(
                await _call(
                    bridge, "chat_folder_create", {"name": "Runs", "project_dir": str(other)}
                )
            )
            out = await _call(bridge, "chat_folder_move", {"folder": "Runs", "new_parent": "Bound"})
        assert out.startswith("Moved folder"), out
        assert _folder(state, runs)["parent_id"] == bound
        assert _resolve_folder_project_dir(state._folders, runs) == (
            os.path.realpath(str(other)),
            None,
        )
