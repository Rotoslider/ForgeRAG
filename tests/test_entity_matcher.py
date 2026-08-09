"""EntityMatcher index-driven fuzzy matching.

The linear O(entities x windows) SequenceMatcher scan covered ~0.007% of a
276k-entity population inside its 5s budget (observed live) — fuzzy
expansion had silently become a lottery. The trigram inverted index must
deliver FULL coverage in milliseconds with unchanged scoring semantics.
"""

import random
import string
import time

from backend.services.entity_matcher import EntityMatcher, _EntityEntry, _normalize


def _matcher(names: list[tuple[str, str]]) -> EntityMatcher:
    m = EntityMatcher(neo4j=None)
    m._entities = [
        _EntityEntry(name=n, normalized=_normalize(n), entity_type=t)
        for n, t in names
    ]
    m._build_index()
    return m


BASE = [
    ("Inconel® 625", "Material"),
    ("Stainless Steel 304", "Material"),
    ("AISI 4340", "Material"),
    ("ASTM A36", "Standard"),
    ("Gas Tungsten Arc Welding", "Process"),
    ("C26000", "Material"),
]


def test_exact_normalized_match_scores_one():
    m = _matcher(BASE)
    got = {r.name: r.score for r in m.find_matches("properties of inconel 625 alloy")}
    assert got.get("Inconel® 625") == 1.0


def test_spacing_and_symbol_variants_match():
    m = _matcher(BASE)
    got = {r.name for r in m.find_matches("what about inconel625 here")}
    assert "Inconel® 625" in got


def test_fuzzy_typo_matches_above_threshold():
    m = _matcher(BASE)
    results = m.find_matches("welding incanel 625 pipe")  # typo: incanel
    got = {r.name: r.score for r in results}
    assert "Inconel® 625" in got
    assert 0.75 <= got["Inconel® 625"] < 1.0


def test_threshold_respected():
    m = _matcher(BASE)
    # 'copper' shares little with any base entity — nothing should match.
    assert m.find_matches("pure copper wire") == []


def test_short_code_entities_match():
    m = _matcher(BASE)
    got = {r.name for r in m.find_matches("yield strength of astm a36 shapes")}
    assert "ASTM A36" in got


def test_full_coverage_at_scale_under_a_second():
    # 120k synthetic entities + planted targets: the linear scan physically
    # could not cover this inside its 5s budget; the index must find every
    # planted target and finish fast.
    rng = random.Random(42)
    synth = [
        ("".join(rng.choices(string.ascii_lowercase, k=rng.randint(6, 14))),
         "Material")
        for _ in range(120_000)
    ]
    planted = [
        ("Inconel® 625", "Material"),
        ("Hastelloy C-276", "Material"),
        ("ASTM A992", "Standard"),
    ]
    # Plant targets at the very END so a budget-limited linear scan would
    # never reach them.
    m = _matcher(synth + planted)

    t0 = time.monotonic()
    results = m.find_matches("compare inconel 625 with hastelloy c276 per astm a992")
    took = time.monotonic() - t0

    names = {r.name for r in results}
    assert {"Inconel® 625", "Hastelloy C-276", "ASTM A992"} <= names
    assert took < 1.0, f"index-driven matching took {took:.2f}s"


def test_lazy_index_build_for_injected_entities():
    m = EntityMatcher(neo4j=None)
    m._entities = [
        _EntityEntry(name="AISI 4140", normalized=_normalize("AISI 4140"),
                     entity_type="Material")
    ]
    # No explicit _build_index — find_matches must build it lazily.
    got = {r.name for r in m.find_matches("hardness of aisi 4140 bar")}
    assert "AISI 4140" in got
