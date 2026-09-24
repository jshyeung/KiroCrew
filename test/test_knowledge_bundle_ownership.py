"""Knowledge bundle round-trip: per-document ownership survives export/import.

Ownership lives in three per-document state tables (``folder_file_state``,
``artifact_item_state``, ``agent_item_state``). An item without its row is
unmanaged: nothing claims it for de-duplication, the Sources UI has no group label
for it, and a later ingest of the same document adds a second copy instead of
replacing the first.

A source's identity across two stores is its ``uri``, not its id -- ``sources.uri``
carries the UNIQUE constraint and each store mints its own ids -- so every row
pointing at a source is rewritten through a uri-keyed map on import.
"""

import json

import pytest

from kiro_crew.dashboard.handlers.knowledge import _validate_knowledge_bundle
from kiro_crew.knowledge.store import (
    BUNDLE_STATE_KEY_COL,
    KnowledgeBundleError,
    KnowledgeStore,
)

ARTIFACT_URI = "artifact://"
AGENT_URI = "agent://"


@pytest.fixture()
def exporter(tmp_path):
    s = KnowledgeStore(str(tmp_path / "exporter.db"))
    yield s
    s.close()


@pytest.fixture()
def importer(tmp_path):
    s = KnowledgeStore(str(tmp_path / "importer.db"))
    yield s
    s.close()


def _artifact_doc(store, *, slug="doc", body="artifact body", name="Doc", uri=ARTIFACT_URI):
    """One owned artifact document: a source, an item, and the ownership row."""
    sid = store.get_source_by_uri(uri)
    sid = sid["id"] if sid else store.add_source(name="Artifacts", source_type="artifact", uri=uri)
    item_id = store.add_item(title=name, content=body, item_type="document", source_id=sid)
    store.db.execute(
        "INSERT OR REPLACE INTO artifact_item_state "
        "(source_id, slug, content_hash, item_ids, updated_at, name, status, kind) "
        "VALUES (?, ?, ?, ?, ?, ?, 'active', 'markdown')",
        (sid, slug, "hash-" + slug, json.dumps([item_id]), "2026-01-01T00:00:00", name),
    )
    store.db.commit()
    return sid, item_id


def _folder_doc(store, *, uri="/remote/notes", file_path="/remote/notes/a.md", body="folder body"):
    sid = store.add_source(name="Notes", source_type="local_folder", uri=uri)
    item_id = store.add_item(title="a.md", content=body, item_type="document", source_id=sid)
    store.db.execute(
        "INSERT INTO folder_file_state (source_id, file_path, content_hash, "
        "text_hash, mtime, item_ids, last_seen, status, attempts) "
        "VALUES (?, ?, 'raw-hash', 'text-hash', 123.5, ?, ?, 'done', 0)",
        (sid, file_path, json.dumps([item_id]), "2026-01-01T00:00:00"),
    )
    store.db.commit()
    return sid, item_id


def _agent_doc(store, *, slug="note", body="agent body", source_uri="https://x/y"):
    sid = store.add_source(name="Auto-added", source_type="agent", uri=AGENT_URI)
    item_id = store.add_item(title="Note", content=body, item_type="document", source_id=sid)
    store.db.execute(
        "INSERT INTO agent_item_state (source_id, slug, content_hash, item_ids, "
        "updated_at, name, status, source_uri) "
        "VALUES (?, ?, 'agent-hash', ?, ?, 'Note', 'active', ?)",
        (sid, slug, json.dumps([item_id]), "2026-01-01T00:00:00", source_uri),
    )
    store.db.commit()
    return sid, item_id


def _owner_of(store, table, item_id):
    """The (source_id, key) whose group holds *item_id*, or None."""
    key_col = BUNDLE_STATE_KEY_COL[table]
    for row in store.db.execute(
        f"SELECT source_id, {key_col} AS k, item_ids FROM {table}"
    ):  # noqa: S608
        if item_id in json.loads(row["item_ids"] or "[]"):
            return row["source_id"], row["k"]
    return None


def _item_count(store, body):
    return sum(1 for r in store.db.execute("SELECT content FROM items") if body in r["content"])


class TestBundleCarriesOwnership:
    """export_all ships the three state tables and import restores them."""

    def test_export_lists_every_state_table(self, exporter):
        _artifact_doc(exporter)
        bundle = exporter.export_all()
        for table in BUNDLE_STATE_KEY_COL:
            assert table in bundle, f"{table} missing from the bundle"

    def test_artifact_ownership_survives_round_trip(self, exporter, importer):
        _, item_id = _artifact_doc(exporter)
        result = importer.import_bundle(exporter.export_all())
        assert result["items_imported"] == 1
        assert result["ownership_rows_imported"] == 1
        owner = _owner_of(importer, "artifact_item_state", item_id)
        assert owner is not None
        local_sid = importer.get_source_by_uri(ARTIFACT_URI)["id"]
        assert owner == (local_sid, "doc")

    def test_folder_and_agent_ownership_survive_round_trip(self, exporter, importer):
        _, folder_item = _folder_doc(exporter)
        _, agent_item = _agent_doc(exporter)
        result = importer.import_bundle(exporter.export_all())
        assert result["ownership_rows_imported"] == 2
        assert _owner_of(importer, "folder_file_state", folder_item) is not None
        assert _owner_of(importer, "agent_item_state", agent_item) is not None

    def test_folder_row_keeps_the_hashes_that_prevent_a_rescan(self, exporter, importer):
        _folder_doc(exporter)
        importer.import_bundle(exporter.export_all())
        row = importer.db.execute(
            "SELECT content_hash, text_hash, status FROM folder_file_state"
        ).fetchone()
        assert row["content_hash"] == "raw-hash"
        assert row["text_hash"] == "text-hash"
        assert row["status"] == "done"

    def test_artifact_row_keeps_hash_name_and_kind(self, exporter, importer):
        _artifact_doc(exporter)
        importer.import_bundle(exporter.export_all())
        row = importer.db.execute(
            "SELECT content_hash, name, kind, status, merged_into_source_id "
            "FROM artifact_item_state"
        ).fetchone()
        assert (row["content_hash"], row["name"], row["kind"]) == ("hash-doc", "Doc", "markdown")
        assert row["status"] == "active"
        assert row["merged_into_source_id"] is None

    def test_agent_row_keeps_its_document_locator(self, exporter, importer):
        _agent_doc(exporter, source_uri="https://example.test/page")
        importer.import_bundle(exporter.export_all())
        row = importer.db.execute("SELECT source_uri FROM agent_item_state").fetchone()
        assert row["source_uri"] == "https://example.test/page"


class TestSourceUriCollision:
    """A uri already present locally holds a DIFFERENT id, and the import has to
    survive that: the source row cannot be inserted, so every pointer into it has
    to be rewritten or the foreign key refuses the whole write."""

    def test_import_succeeds_when_the_uri_is_already_local(self, exporter, importer):
        _artifact_doc(exporter, slug="remote", body="remote body", name="Remote")
        _artifact_doc(importer, slug="local", body="local body", name="Local")
        remote_sid = exporter.get_source_by_uri(ARTIFACT_URI)["id"]
        local_sid = importer.get_source_by_uri(ARTIFACT_URI)["id"]
        assert remote_sid != local_sid

        result = importer.import_bundle(exporter.export_all())

        assert result["items_imported"] == 1
        assert _item_count(importer, "remote body") == 1
        assert _item_count(importer, "local body") == 1

    def test_collision_import_files_items_under_the_local_source(self, exporter, importer):
        _artifact_doc(exporter, slug="remote", body="remote body", name="Remote")
        _artifact_doc(importer, slug="local", body="local body", name="Local")
        local_sid = importer.get_source_by_uri(ARTIFACT_URI)["id"]

        importer.import_bundle(exporter.export_all())

        source_ids = {r["source_id"] for r in importer.db.execute("SELECT source_id FROM items")}
        assert source_ids == {local_sid}
        known = {r["id"] for r in importer.db.execute("SELECT id FROM sources")}
        assert source_ids <= known, "an item points at a source row that is absent"

    def test_collision_import_restores_ownership_for_the_imported_document(
        self, exporter, importer
    ):
        _, remote_item = _artifact_doc(exporter, slug="remote", body="remote body", name="Remote")
        _artifact_doc(importer, slug="local", body="local body", name="Local")
        local_sid = importer.get_source_by_uri(ARTIFACT_URI)["id"]

        result = importer.import_bundle(exporter.export_all())

        assert result["ownership_rows_imported"] == 1
        assert _owner_of(importer, "artifact_item_state", remote_item) == (local_sid, "remote")
        slugs = {r["slug"] for r in importer.db.execute("SELECT slug FROM artifact_item_state")}
        assert slugs == {"local", "remote"}

    def test_reimporting_the_same_bundle_changes_nothing(self, exporter, importer):
        _artifact_doc(exporter)
        bundle = exporter.export_all()
        importer.import_bundle(bundle)
        second = importer.import_bundle(bundle)
        assert second["items_imported"] == 0
        assert second["ownership_rows_imported"] == 0
        assert _item_count(importer, "artifact body") == 1
        assert (
            importer.db.execute("SELECT COUNT(*) AS n FROM artifact_item_state").fetchone()["n"]
            == 1
        )

    def test_exporting_store_can_reimport_its_own_bundle(self, exporter):
        _, item_id = _artifact_doc(exporter)
        result = exporter.import_bundle(exporter.export_all())
        assert result["items_imported"] == 0
        assert _item_count(exporter, "artifact body") == 1
        assert _owner_of(exporter, "artifact_item_state", item_id) is not None


class TestUriAbsentLocally:
    """A bundle uri this store does not hold is CREATED, so its documents arrive
    owned rather than being dropped."""

    def test_absent_uri_creates_the_source(self, exporter, importer):
        _folder_doc(exporter, uri="/remote/notes")
        importer.import_bundle(exporter.export_all())
        created = importer.get_source_by_uri("/remote/notes")
        assert created is not None
        assert created["source_type"] == "local_folder"

    def test_absent_uri_keeps_the_bundle_source_id(self, exporter, importer):
        remote_sid, _ = _folder_doc(exporter, uri="/remote/notes")
        importer.import_bundle(exporter.export_all())
        assert importer.get_source_by_uri("/remote/notes")["id"] == remote_sid

    def test_absent_uri_whose_id_is_taken_gets_a_fresh_id(self, exporter, importer):
        remote_sid, _ = _folder_doc(exporter, uri="/remote/notes")
        # Same id locally, different uri: the id cannot be reused and the uri is
        # still absent, so neither reusing nor skipping is right.
        importer.db.execute(
            "INSERT INTO sources (id, name, source_type, uri, properties, "
            "sync_status, created_at, updated_at) "
            "VALUES (?, 'Other', 'local_folder', '/local/other', '{}', 'paused', "
            "'2026-01-01T00:00:00', '2026-01-01T00:00:00')",
            (remote_sid,),
        )
        importer.db.commit()

        result = importer.import_bundle(exporter.export_all())

        created = importer.get_source_by_uri("/remote/notes")
        assert created is not None
        assert created["id"] != remote_sid
        assert result["items_imported"] == 1
        assert result["ownership_rows_imported"] == 1
        assert _owner_of(
            importer, "folder_file_state", next(iter(_imported_item_ids(importer, "folder body")))
        )
        assert importer.get_source_by_uri("/local/other")["name"] == "Other"

    def test_walking_source_still_waits_for_confirmation(self, exporter, importer):
        _folder_doc(exporter, uri="/remote/notes")
        importer.import_bundle(exporter.export_all())
        row = importer.get_source_by_uri("/remote/notes")
        assert row["sync_status"] == "pending_confirmation"


def _imported_item_ids(store, body):
    return {
        r["id"] for r in store.db.execute("SELECT id, content FROM items") if body in r["content"]
    }


class TestOwnershipIsOnlyRestoredForItemsThatArrived:
    """A state row names a group. Restoring one whose group is not fully here would
    let a later ingest replace part of the document and duplicate the rest."""

    def test_group_naming_an_item_outside_the_bundle_is_dropped(self, exporter, importer):
        _artifact_doc(exporter)
        bundle = exporter.export_all()
        bundle["artifact_item_state"][0]["item_ids"] = json.dumps(["not-in-this-bundle"])
        result = importer.import_bundle(bundle)
        assert result["items_imported"] == 1
        assert result["ownership_rows_imported"] == 0

    def test_group_is_dropped_when_one_item_lands_under_another_source(self, exporter, importer):
        """An item id already present here keeps its LOCAL source, so a group naming
        it is not fully held by the source the state row names -- even though every
        id in the group is in the bundle."""
        other = importer.add_source(name="Other", source_type="local_file", uri="/local/other.md")
        importer.db.execute(
            "INSERT INTO items (id, title, content, item_type, source_id, "
            "created_at, updated_at) VALUES ('shared-id', 'Local', 'local text', "
            "'document', ?, '2026-01-01T00:00:00', '2026-01-01T00:00:00')",
            (other,),
        )
        importer.db.commit()

        _, first = _artifact_doc(exporter)
        bundle = exporter.export_all()
        remote_sid = exporter.get_source_by_uri(ARTIFACT_URI)["id"]
        bundle["items"].append(
            {
                "id": "shared-id",
                "title": "Second",
                "content": "second chunk",
                "item_type": "document",
                "source_id": remote_sid,
            }
        )
        bundle["artifact_item_state"][0]["item_ids"] = json.dumps([first, "shared-id"])

        result = importer.import_bundle(bundle)

        # Only the first item lands; 'shared-id' is already taken locally.
        assert result["items_imported"] == 1
        assert result["ownership_rows_imported"] == 0
        assert (
            importer.db.execute("SELECT source_id FROM items WHERE id = 'shared-id'").fetchone()[
                "source_id"
            ]
            == other
        )

    def test_group_cannot_claim_items_the_bundle_did_not_ship(self, exporter, importer):
        """A bundle naming ids that exist LOCALLY must not take them over: the
        local document's own row would then be one of two naming the same items."""
        _, local_item = _artifact_doc(importer, slug="local", body="local body", name="Local")
        _artifact_doc(exporter, slug="remote", body="remote body", name="Remote")
        bundle = exporter.export_all()
        bundle["artifact_item_state"][0]["item_ids"] = json.dumps([local_item])

        result = importer.import_bundle(bundle)

        assert result["ownership_rows_imported"] == 0
        assert _owner_of(importer, "artifact_item_state", local_item) == (
            importer.get_source_by_uri(ARTIFACT_URI)["id"],
            "local",
        )

    def test_a_group_cannot_adopt_unowned_local_content(self, exporter, importer):
        """Residue -- an item under the mapped source that no state row owns -- is
        exactly what a bundle must not be able to claim. It is content this store
        already holds and the bundle never shipped, and adopting it would hand a
        foreign document the power to replace or delete it."""
        local_sid = importer.add_source(name="Artifacts", source_type="artifact", uri=ARTIFACT_URI)
        importer.db.execute(
            "INSERT INTO items (id, title, content, item_type, source_id, "
            "created_at, updated_at) VALUES ('residue-id', 'Residue', "
            "'residue text', 'document', ?, '2026-01-01T00:00:00', "
            "'2026-01-01T00:00:00')",
            (local_sid,),
        )
        importer.db.commit()
        _artifact_doc(exporter, slug="theirs", name="Theirs")
        bundle = exporter.export_all()
        bundle["artifact_item_state"][0]["item_ids"] = json.dumps(["residue-id"])

        result = importer.import_bundle(bundle)

        assert result["ownership_rows_imported"] == 0
        assert (
            importer.db.execute("SELECT COUNT(*) AS n FROM artifact_item_state").fetchone()["n"]
            == 0
        )

    def test_empty_group_is_dropped(self, exporter, importer):
        _artifact_doc(exporter)
        bundle = exporter.export_all()
        bundle["artifact_item_state"][0]["item_ids"] = "[]"
        assert importer.import_bundle(bundle)["ownership_rows_imported"] == 0

    def test_unparsable_group_is_dropped(self, exporter, importer):
        _artifact_doc(exporter)
        bundle = exporter.export_all()
        bundle["artifact_item_state"][0]["item_ids"] = "{not json"
        assert importer.import_bundle(bundle)["ownership_rows_imported"] == 0

    def test_local_row_for_the_same_document_wins(self, exporter, importer):
        """Same (source uri, slug) on both sides: replacing the local row would
        leave the items it owns with nothing naming them."""
        _, local_item = _artifact_doc(importer, slug="doc", body="local body", name="Local")
        _artifact_doc(exporter, slug="doc", body="remote body", name="Remote")

        result = importer.import_bundle(exporter.export_all())

        assert result["ownership_rows_imported"] == 0
        assert _owner_of(importer, "artifact_item_state", local_item) is not None
        row = importer.db.execute(
            "SELECT name FROM artifact_item_state WHERE slug = 'doc'"
        ).fetchone()
        assert row["name"] == "Local"


class TestAKeyThisStoreAlreadyHoldsBlocksTheItems:
    """A `(source, key)` pair IS a document's identity within a source, so two
    documents cannot share one. When a live local row holds the pair, the bundle's
    copy is not a document here and its items must not arrive: nothing would ever
    own them, and the text would answer searches beside the local copy forever."""

    def test_the_colliding_items_do_not_arrive(self, exporter, importer):
        _artifact_doc(importer, slug="doc", body="local body", name="Local")
        _artifact_doc(exporter, slug="doc", body="remote body", name="Remote")

        result = importer.import_bundle(exporter.export_all())

        assert result["items_imported"] == 0
        assert _item_count(importer, "remote body") == 0
        assert _item_count(importer, "local body") == 1

    def test_no_item_is_left_without_an_owner(self, exporter, importer):
        _artifact_doc(importer, slug="doc", body="local body", name="Local")
        _artifact_doc(exporter, slug="doc", body="remote body", name="Remote")

        importer.import_bundle(exporter.export_all())

        owned = set()
        for table in BUNDLE_STATE_KEY_COL:
            for row in importer.db.execute(f"SELECT item_ids FROM {table}"):  # noqa: S608
                owned.update(json.loads(row["item_ids"] or "[]"))
        present = {r["id"] for r in importer.db.execute("SELECT id FROM items")}
        assert present == owned, "an item arrived that no ownership row names"

    def test_an_empty_local_row_does_not_block_its_items(self, exporter, importer):
        """Only a LIVE local row blocks. An empty group is a marker, and the
        imported document takes the key instead."""
        _artifact_doc(exporter, slug="doc", body="remote body", name="Remote")
        local_sid = importer.add_source(name="Artifacts", source_type="artifact", uri=ARTIFACT_URI)
        importer.db.execute(
            "INSERT INTO artifact_item_state (source_id, slug, content_hash, "
            "item_ids, updated_at, name, status) VALUES (?, 'doc', 'loser-hash', "
            "'[]', '2026-01-01T00:00:00', 'Local', 'deduped')",
            (local_sid,),
        )
        importer.db.commit()

        result = importer.import_bundle(exporter.export_all())

        assert result["items_imported"] == 1
        assert _item_count(importer, "remote body") == 1

    def test_a_document_this_store_does_not_have_still_arrives(self, exporter, importer):
        _artifact_doc(importer, slug="local", body="local body", name="Local")
        _artifact_doc(exporter, slug="remote", body="remote body", name="Remote")

        result = importer.import_bundle(exporter.export_all())

        assert result["items_imported"] == 1
        assert result["ownership_rows_imported"] == 1


class TestOneItemIsNeverOwnedByTwoRows:
    """The next document-level delete hands `delete_items_batch` an item whose only
    other holder is a state row it does not consult, finds nothing else holding it,
    and removes it -- leaving the second row naming deleted content."""

    def test_overlapping_groups_within_one_bundle_yield_one_owner(self, exporter, importer):
        _, item_id = _artifact_doc(exporter, slug="first", name="First")
        bundle = exporter.export_all()
        first = bundle["artifact_item_state"][0]
        bundle["artifact_item_state"].append({**first, "slug": "second", "name": "Second"})

        result = importer.import_bundle(bundle)

        assert result["ownership_rows_imported"] == 1
        owners = [
            row["slug"]
            for row in importer.db.execute("SELECT slug, item_ids FROM artifact_item_state")
            if item_id in json.loads(row["item_ids"] or "[]")
        ]
        assert owners == ["first"]

    def test_a_group_an_existing_row_already_owns_is_refused(self, exporter, importer):
        """The same item id under a DIFFERENT document key: the local row owns it,
        so the imported row may not name it as well."""
        local_sid = importer.add_source(name="Artifacts", source_type="artifact", uri=ARTIFACT_URI)
        importer.db.execute(
            "INSERT INTO items (id, title, content, item_type, source_id, "
            "created_at, updated_at) VALUES ('shared-id', 'Local', 'shared text', "
            "'document', ?, '2026-01-01T00:00:00', '2026-01-01T00:00:00')",
            (local_sid,),
        )
        importer.db.execute(
            "INSERT INTO artifact_item_state (source_id, slug, content_hash, "
            "item_ids, updated_at, name, status) VALUES (?, 'mine', 'h', "
            "'[\"shared-id\"]', '2026-01-01T00:00:00', 'Mine', 'active')",
            (local_sid,),
        )
        importer.db.commit()
        _artifact_doc(exporter, slug="theirs", name="Theirs")
        bundle = exporter.export_all()
        bundle["items"].append(
            {
                "id": "shared-id",
                "title": "Theirs",
                "content": "shared text",
                "item_type": "document",
                "source_id": exporter.get_source_by_uri(ARTIFACT_URI)["id"],
            }
        )
        bundle["artifact_item_state"][0]["item_ids"] = json.dumps(["shared-id"])

        result = importer.import_bundle(bundle)

        assert result["ownership_rows_imported"] == 0
        owners = [
            row["slug"]
            for row in importer.db.execute("SELECT slug, item_ids FROM artifact_item_state")
            if "shared-id" in json.loads(row["item_ids"] or "[]")
        ]
        assert owners == ["mine"]


class TestIdentityColumnsMustBeText:
    """Both become dictionary keys while the bundle's source ids are rewritten, so a
    list or dict raises an unhashable-type TypeError that no handler arm catches --
    a 500 in place of the typed 400. Enforced at the writer, like the properties and
    aliases columns, so a caller that is not the dashboard endpoint is safe too."""

    @pytest.mark.parametrize("bad", [[], {}, 7, None, ""])
    def test_a_source_id_that_is_not_text_is_refused(self, importer, bad):
        bundle = {
            "sources": [{"id": bad, "name": "A", "source_type": "local_file", "uri": "/a.md"}]
        }
        with pytest.raises(KnowledgeBundleError, match="sources.id"):
            importer.import_bundle(bundle)

    @pytest.mark.parametrize("bad", [[], {}, 7, None, ""])
    def test_a_source_uri_that_is_not_text_is_refused(self, importer, bad):
        bundle = {"sources": [{"id": "s1", "name": "A", "source_type": "local_file", "uri": bad}]}
        with pytest.raises(KnowledgeBundleError, match="sources.uri"):
            importer.import_bundle(bundle)

    def test_a_state_row_source_id_that_is_not_text_is_skipped_not_raised(self, exporter, importer):
        _artifact_doc(exporter)
        bundle = exporter.export_all()
        bundle["artifact_item_state"][0]["source_id"] = []

        result = importer.import_bundle(bundle)

        assert result["items_imported"] == 1
        assert result["ownership_rows_imported"] == 0


class TestUnresolvablePointersAreStillRefused:
    """The remap rewrites a pointer it can resolve and leaves one it cannot exactly
    as it arrived, so a bundle naming a source that is nowhere keeps being refused
    by the foreign key instead of being filed somewhere plausible."""

    def test_item_naming_an_unknown_source_is_refused(self, importer):
        bundle = {
            "sources": [],
            "items": [
                {
                    "id": "i1",
                    "title": "T",
                    "content": "orphan text",
                    "item_type": "document",
                    "source_id": "no-such-source",
                }
            ],
        }
        with pytest.raises(Exception, match="FOREIGN KEY"):
            importer.import_bundle(bundle)
        assert importer.db.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"] == 0

    def test_location_naming_an_unknown_source_is_refused(self, importer):
        bundle = {
            "source_locations": [
                {"id": "loc1", "item_id": "no-such-item", "source_id": "no-such-source"}
            ]
        }
        with pytest.raises(Exception, match="FOREIGN KEY"):
            importer.import_bundle(bundle)

    def test_item_with_no_source_at_all_still_imports(self, importer):
        bundle = {
            "items": [{"id": "i1", "title": "T", "content": "loose text", "item_type": "document"}]
        }
        assert importer.import_bundle(bundle)["items_imported"] == 1
        assert (
            importer.db.execute("SELECT source_id FROM items WHERE id = 'i1'").fetchone()[
                "source_id"
            ]
            is None
        )


class TestBundleValidation:
    """The new lists go through the same shape checks as the existing ones."""

    @pytest.mark.parametrize("table", sorted(BUNDLE_STATE_KEY_COL))
    def test_state_table_must_be_a_list(self, table):
        assert _validate_knowledge_bundle({table: {}}) == f"'{table}' must be a list"

    @pytest.mark.parametrize("table", sorted(BUNDLE_STATE_KEY_COL))
    def test_state_entries_must_be_objects(self, table):
        assert _validate_knowledge_bundle({table: ["nope"]}) == f"'{table}' entries must be objects"

    def test_key_column_must_be_text(self):
        assert (
            _validate_knowledge_bundle({"artifact_item_state": [{"slug": 7}]})
            == "'artifact_item_state.slug' must be a string or null"
        )

    def test_display_name_must_be_text(self):
        assert (
            _validate_knowledge_bundle({"artifact_item_state": [{"name": 7}]})
            == "'artifact_item_state.name' must be a string or null"
        )

    def test_mtime_shape_is_not_validated_because_it_does_not_travel(self):
        assert _validate_knowledge_bundle({"folder_file_state": [{"mtime": "soon"}]}) is None

    def test_numeric_mtime_and_absent_tables_pass(self):
        assert _validate_knowledge_bundle({"folder_file_state": [{"mtime": 1.5}]}) is None
        assert _validate_knowledge_bundle({"items": []}) is None


class TestHostLocalMtimeDoesNotTravel:
    """The folder scan skips hashing a live row whose recorded mtime is at or above
    the file's. A carried mtime from the exporting host would therefore stop the
    local file being hashed again, leaving the imported text indexed in its place."""

    def test_imported_folder_row_records_a_floor_mtime(self, exporter, importer):
        _folder_doc(exporter)
        exporter.db.execute("UPDATE folder_file_state SET mtime = 99999999999.0")
        exporter.db.commit()

        importer.import_bundle(exporter.export_all())

        assert importer.db.execute("SELECT mtime FROM folder_file_state").fetchone()["mtime"] == 0.0

    def test_the_floor_is_below_any_real_file_timestamp(self, exporter, importer):
        """The gate is `mtime <= state['mtime']`, so the stored value has to be under
        a real timestamp for the first local scan to hash the file at all."""
        import time

        _folder_doc(exporter)
        importer.import_bundle(exporter.export_all())
        stored = importer.db.execute("SELECT mtime FROM folder_file_state").fetchone()["mtime"]
        assert stored < time.time()

    def test_the_content_hashes_still_travel(self, exporter, importer):
        """They are what keeps the re-hash cheap: the scan hashes the file, matches,
        and skips extraction."""
        _folder_doc(exporter)
        importer.import_bundle(exporter.export_all())
        row = importer.db.execute(
            "SELECT content_hash, text_hash FROM folder_file_state"
        ).fetchone()
        assert (row["content_hash"], row["text_hash"]) == ("raw-hash", "text-hash")


class TestLocalRowWinsOnlyWhileItOwnsItems:
    """A local row with an empty or stale group holds a marker, not ownership, and
    nothing else will ever name the items the bundle brought."""

    def test_empty_local_group_yields_to_the_imported_row(self, exporter, importer):
        _, remote_item = _artifact_doc(exporter, slug="doc", body="remote body", name="Remote")
        local_sid = importer.add_source(name="Artifacts", source_type="artifact", uri=ARTIFACT_URI)
        # A document that lost a de-duplication: a live row with no group.
        importer.db.execute(
            "INSERT INTO artifact_item_state (source_id, slug, content_hash, "
            "item_ids, updated_at, name, status) VALUES (?, 'doc', 'loser-hash', "
            "'[]', '2026-01-01T00:00:00', 'Local', 'deduped')",
            (local_sid,),
        )
        importer.db.commit()

        result = importer.import_bundle(exporter.export_all())

        assert result["ownership_rows_imported"] == 1
        assert _owner_of(importer, "artifact_item_state", remote_item) == (local_sid, "doc")

    def test_stale_local_group_yields_to_the_imported_row(self, exporter, importer):
        _, remote_item = _artifact_doc(exporter, slug="doc", body="remote body", name="Remote")
        local_sid = importer.add_source(name="Artifacts", source_type="artifact", uri=ARTIFACT_URI)
        # A group naming only items that have since been deleted.
        importer.db.execute(
            "INSERT INTO artifact_item_state (source_id, slug, content_hash, "
            "item_ids, updated_at, name, status) VALUES (?, 'doc', 'gone-hash', "
            "'[\"deleted-item\"]', '2026-01-01T00:00:00', 'Local', 'active')",
            (local_sid,),
        )
        importer.db.commit()

        result = importer.import_bundle(exporter.export_all())

        assert result["ownership_rows_imported"] == 1
        assert _owner_of(importer, "artifact_item_state", remote_item) is not None

    def test_displaced_marker_releases_its_claim_on_another_source_items(self, exporter, importer):
        """Leaving the claim behind under a hash no row names means a later deletion
        of the holder reassigns an item here with nothing to adopt it into."""
        _artifact_doc(exporter, slug="doc", body="remote body", name="Remote")
        winner = importer.add_source(
            name="Winner", source_type="local_file", uri="/local/winner.md"
        )
        held = importer.add_item(
            title="Winner",
            content="shared text",
            item_type="document",
            source_id=winner,
            content_hash="loser-hash",
        )
        local_sid = importer.add_source(name="Artifacts", source_type="artifact", uri=ARTIFACT_URI)
        importer.db.execute(
            "INSERT INTO artifact_item_state (source_id, slug, content_hash, "
            "item_ids, updated_at, name, status) VALUES (?, 'doc', 'loser-hash', "
            "'[]', '2026-01-01T00:00:00', 'Local', 'deduped')",
            (local_sid,),
        )
        importer.add_source_location(held, local_sid)
        importer.db.commit()
        assert (
            importer.db.execute(
                "SELECT COUNT(*) AS n FROM source_locations WHERE source_id = ?", (local_sid,)
            ).fetchone()["n"]
            == 1
        )

        importer.import_bundle(exporter.export_all())

        assert (
            importer.db.execute(
                "SELECT COUNT(*) AS n FROM source_locations WHERE source_id = ?", (local_sid,)
            ).fetchone()["n"]
            == 0
        )
        assert (
            importer.db.execute("SELECT COUNT(*) AS n FROM items WHERE id = ?", (held,)).fetchone()[
                "n"
            ]
            == 1
        ), "the winner's item must not be deleted"


class TestDuplicateSourceIdIsMalformed:
    """One id may name only one uri. ``sources.id`` is a PRIMARY KEY, so no export
    produces a repeat; accepting one would let the later entry overwrite the earlier
    one's place in the map and file items under a source never named for them."""

    def test_repeated_id_under_two_uris_is_refused(self, importer):
        bundle = {
            "sources": [
                {"id": "s1", "name": "A", "source_type": "local_file", "uri": "/a.md"},
                {"id": "s1", "name": "B", "source_type": "local_file", "uri": "/b.md"},
            ]
        }
        with pytest.raises(KnowledgeBundleError, match="repeats an id"):
            importer.import_bundle(bundle)
        assert importer.db.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"] == 0

    def test_repeated_id_under_the_same_uri_is_fine(self, importer):
        bundle = {
            "sources": [
                {"id": "s1", "name": "A", "source_type": "local_file", "uri": "/a.md"},
                {"id": "s1", "name": "A", "source_type": "local_file", "uri": "/a.md"},
            ]
        }
        importer.import_bundle(bundle)
        assert importer.db.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"] == 1


class TestLegacyBundle:
    """A bundle written without the state tables imports exactly as before."""

    def test_missing_state_tables_import_cleanly(self, exporter, importer):
        _artifact_doc(exporter)
        bundle = exporter.export_all()
        for table in BUNDLE_STATE_KEY_COL:
            bundle.pop(table)
        result = importer.import_bundle(bundle)
        assert result["items_imported"] == 1
        assert result["ownership_rows_imported"] == 0
