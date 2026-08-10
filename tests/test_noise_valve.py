"""N2 extraction-time noise valve.

Blocklisted generic nouns reroute to topic_tags instead of becoming
entity nodes; relationships must resolve against what the page actually
extracted (previously a silent MATCH drop in graph_builder, and a
hallucinated link to an off-page entity would even succeed); generic
single words are flagged for the next N1 round but never dropped on
wordform alone.
"""

import backend.ingestion.noise_valve as nv
from backend.ingestion.entity_extractor import (
    EquipmentMention,
    FormulaMention,
    MaterialMention,
    PageExtraction,
    ProcessMention,
    Relationship,
    StandardMention,
)
from backend.ingestion.noise_valve import apply_noise_valve


def _blocklist(monkeypatch, table):
    monkeypatch.setattr(nv, "_blocklist_cache", table)


def _extraction(**kw):
    return PageExtraction(**kw)


def test_blocklisted_material_rerouted_to_topic_tags(monkeypatch):
    _blocklist(monkeypatch, {"Material": {"steel"}})
    ex = _extraction(
        materials=[MaterialMention(name="Steel"),
                   MaterialMention(name="AISI 4140")],
        topic_tags=["heat-treatment"],
    )
    out, report = apply_noise_valve(ex)
    assert [m.name for m in out.materials] == ["AISI 4140"]
    assert report.rerouted == [("Material", "Steel")]
    assert out.topic_tags == ["heat-treatment", "steel"]


def test_designations_pass_untouched(monkeypatch):
    _blocklist(monkeypatch, {"Material": {"steel"}, "Process": {"welding"}})
    ex = _extraction(
        materials=[MaterialMention(name="Inconel 625")],
        processes=[ProcessMention(name="GTAW")],
        relationships=[Relationship(
            type="material_compatible_with_process",
            subject="Inconel 625", object="GTAW")],
    )
    out, report = apply_noise_valve(ex)
    assert not report.acted
    assert len(out.relationships) == 1


def test_rel_to_rerouted_entity_dropped_with_reason(monkeypatch):
    _blocklist(monkeypatch, {"Process": {"welding"}})
    ex = _extraction(
        materials=[MaterialMention(name="A36")],
        processes=[ProcessMention(name="welding")],
        relationships=[Relationship(
            type="material_compatible_with_process",
            subject="A36", object="welding")],
    )
    out, report = apply_noise_valve(ex)
    assert out.relationships == []
    assert len(report.dropped_rels) == 1
    assert "rerouted to topic_tags" in report.dropped_rels[0][3]


def test_rel_to_offpage_entity_dropped(monkeypatch):
    _blocklist(monkeypatch, {})
    ex = _extraction(
        materials=[MaterialMention(name="A36")],
        relationships=[Relationship(
            type="material_governed_by_standard",
            subject="A36", object="ASTM A36")],  # standard never extracted
    )
    out, report = apply_noise_valve(ex)
    assert out.relationships == []
    assert "not extracted on page" in report.dropped_rels[0][3]


def test_page_rel_validates_object_only(monkeypatch):
    # mentions_* rels carry the page as implicit subject — the model often
    # writes junk there; only the object must resolve.
    _blocklist(monkeypatch, {})
    ex = _extraction(
        equipment=[EquipmentMention(name="lathe")],
        relationships=[Relationship(
            type="mentions_equipment", subject="page", object="lathe")],
    )
    out, report = apply_noise_valve(ex)
    assert len(out.relationships) == 1
    assert not report.dropped_rels


def test_aliases_and_secondary_ids_resolve_endpoints(monkeypatch):
    _blocklist(monkeypatch, {})
    ex = _extraction(
        materials=[MaterialMention(
            name="Type 304", uns_number="S30400",
            common_names=["304 stainless"])],
        standards=[StandardMention(code="ASTM A240", number="A240")],
        relationships=[
            Relationship(type="material_governed_by_standard",
                         subject="304 stainless", object="A240"),
            Relationship(type="mentions_material", subject="p",
                         object="S30400"),
        ],
    )
    out, report = apply_noise_valve(ex)
    assert len(out.relationships) == 2
    assert not report.dropped_rels


def test_formula_endpoints_resolve(monkeypatch):
    _blocklist(monkeypatch, {})
    ex = _extraction(
        materials=[MaterialMention(name="A36")],
        formulas=[FormulaMention(name="beam deflection")],
        relationships=[Relationship(
            type="formula_uses_material",
            subject="beam deflection", object="A36")],
    )
    out, report = apply_noise_valve(ex)
    assert len(out.relationships) == 1


def test_generic_candidate_flagged_but_kept(monkeypatch):
    _blocklist(monkeypatch, {"Equipment": set()})
    ex = _extraction(equipment=[EquipmentMention(name="bracket"),
                                EquipmentMention(name="CNC VF-2")])
    out, report = apply_noise_valve(ex)
    assert [e.name for e in out.equipment] == ["bracket", "CNC VF-2"]
    assert ("Equipment", "bracket") in report.generic_candidates
    assert all(n != "CNC VF-2" for _, n in report.generic_candidates)


def test_pipeline_applies_valve():
    # The valve only protects future ingests if the pipeline actually
    # calls it after every extraction — pin both call sites.
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "backend" / "ingestion" / "pipeline.py").read_text()
    assert src.count("apply_noise_valve(") >= 2, (
        "both pipeline extraction lanes must pass through the noise valve"
    )


def test_real_blocklist_loads():
    nv._blocklist_cache = None
    try:
        table = nv.get_blocklist()
        assert len(table.get("Material", set())) >= 30
        assert "steel" in table["Material"]
        assert "welding" in table["Process"]
    finally:
        nv._blocklist_cache = None
