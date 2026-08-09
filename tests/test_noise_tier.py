"""N1 stop-tier noise exclusion.

The bulk-era extractor emitted generic-noun entities ("steel": 5,556 pages)
that made graph_first drag unrelated documents into specific queries. The
N1 review marked 92 entities noise_tier='stop' — real entities, kept in
the graph, but excluded from query expansion and graph seeding. Three
consumers must honor the tier; these tests pin the guard into each query
so a refactor can't silently drop it (same doctrine as the JobStep grep
test: the string IS the contract).
"""

import inspect
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

STOP_GUARD = "coalesce(e.noise_tier, '') <> 'stop'"


def test_matcher_refresh_excludes_stop_tier():
    from backend.services.entity_matcher import EntityMatcher
    src = inspect.getsource(EntityMatcher.refresh)
    assert STOP_GUARD in src, (
        "EntityMatcher.refresh must exclude noise_tier='stop' entities — "
        "otherwise stop-tier names re-enter fuzzy query expansion."
    )


def test_graph_strategies_exclude_stop_tier():
    src = (REPO / "backend" / "routers" / "search.py").read_text()
    # graph_first seeds from the entity fulltext index; graph_boosted
    # counts entity matches as score boosts. Both must carry the guard.
    assert src.count(STOP_GUARD) >= 2, (
        "graph_first seeding and graph_boosted boosting must both exclude "
        f"noise_tier='stop' (found {src.count(STOP_GUARD)} guard(s), need 2)."
    )


def test_blocklist_is_banked_and_well_formed():
    path = REPO / "backend" / "resources" / "noise_blocklist.json"
    data = json.loads(path.read_text())
    entries = data["stop_tier"]
    assert len(entries) == 92
    keys = {(e["label"], e["name"]) for e in entries}
    assert len(keys) == len(entries), "duplicate blocklist entries"
    valid_labels = {"Material", "Process", "Standard", "Equipment"}
    assert {e["label"] for e in entries} <= valid_labels
    # The vetoed DELETEs must be present as stop-tier (the veto decision:
    # nothing was deleted; everything was downgraded to reversible marks).
    for pair in [("Material", "water"), ("Material", "resistor"),
                 ("Process", "annealed")]:
        assert pair in keys, f"vetoed-DELETE entry missing: {pair}"
