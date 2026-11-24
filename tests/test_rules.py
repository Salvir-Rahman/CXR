"""Unit tests for the auditable neuro-symbolic rule registry and Horn clause engine."""

import json
from pathlib import Path

import numpy as np
import pytest

from cxr_triage.config import NIH_CHESTXRAY14_LABELS, load_config
from cxr_triage.rules.engine import (
    ClauseAtom,
    FormalClause,
    SymbolicRule,
    compute_normalized_entropy,
    evaluate_rules,
    load_rule_registry,
)


def test_load_default_rule_registry() -> None:
    """Verify loading configs/rules-v1.0.json against the 14 NIH labels."""
    config = load_config()
    rules_path = config.paths.root_dir / "configs" / "rules-v1.0.json"

    rules, sha256_hash = load_rule_registry(rules_path, NIH_CHESTXRAY14_LABELS)

    assert len(rules) >= 6
    assert isinstance(sha256_hash, str) and len(sha256_hash) == 64

    rule_ids = [r.rule_id for r in rules]
    assert "R1.1" in rule_ids
    assert "R1.2" in rule_ids
    assert "R1.3" in rule_ids
    assert "R2.1" in rule_ids
    assert "R2.2" in rule_ids
    assert "R2.3" in rule_ids

    # Verify each rule is immutable and contains only valid NIH labels
    for rule in rules:
        assert isinstance(rule, SymbolicRule)
        assert rule.rule_id.startswith(("R1.", "R2."))
        assert rule.source_guideline.strip() != ""
        assert rule.guideline_quote.strip() != ""
        assert rule.date_introduced.strip() != ""

        clause = rule.formal_clause
        assert isinstance(clause, FormalClause)
        for ant in clause.antecedents:
            assert ant.finding in NIH_CHESTXRAY14_LABELS
            assert ant.state in {"absent", "present", "severe"}

        assert clause.consequent.finding in NIH_CHESTXRAY14_LABELS
        assert clause.consequent.state in {"absent", "present", "severe"}


def test_out_of_ontology_finding_rejected(tmp_path: Path) -> None:
    """Verify registry loader raises ValueError on out-of-ontology labels."""
    bad_registry = {
        "rules": [
            {
                "rule_id": "R1.1",
                "source_guideline": "Test Guideline",
                "guideline_quote": "Test quote",
                "formal_clause": {
                    "antecedents": [{"finding": "COVID-19", "state": "present"}],
                    "consequent": {"finding": "Pneumonia", "state": "present"},
                },
                "date_introduced": "2026-11-24",
            }
        ]
    }
    file_path = tmp_path / "bad_rules.json"
    file_path.write_text(json.dumps(bad_registry), encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-ontology antecedent finding: 'COVID-19'"):
        load_rule_registry(file_path, NIH_CHESTXRAY14_LABELS)


def test_invalid_state_rejected(tmp_path: Path) -> None:
    """Verify registry loader rejects non-canonical severity states."""
    bad_registry = {
        "rules": [
            {
                "rule_id": "R1.1",
                "source_guideline": "Test Guideline",
                "guideline_quote": "Test quote",
                "formal_clause": {
                    "antecedents": [{"finding": "Cardiomegaly", "state": "moderate"}],
                    "consequent": {"finding": "Effusion", "state": "present"},
                },
                "date_introduced": "2026-11-24",
            }
        ]
    }
    file_path = tmp_path / "bad_state.json"
    file_path.write_text(json.dumps(bad_registry), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid state 'moderate'"):
        load_rule_registry(file_path, NIH_CHESTXRAY14_LABELS)


def test_missing_required_fields_rejected(tmp_path: Path) -> None:
    """Verify registry loader enforces the 5-field schema."""
    bad_registry = {
        "rules": [
            {
                "rule_id": "R1.1",
                "source_guideline": "Test Guideline",
                # missing guideline_quote, formal_clause, date_introduced
            }
        ]
    }
    file_path = tmp_path / "missing_fields.json"
    file_path.write_text(json.dumps(bad_registry), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_rule_registry(file_path, NIH_CHESTXRAY14_LABELS)


def test_rule_evaluation_logic() -> None:
    """Verify Horn clause evaluation semantics (supporting, violated, inactive)."""
    rule1 = SymbolicRule(
        rule_id="R1.1",
        source_guideline="ACR CHF",
        guideline_quote="Cardiomegaly and Edema imply Effusion.",
        formal_clause=FormalClause(
            antecedents=(
                ClauseAtom(finding="Cardiomegaly", state="present"),
                ClauseAtom(finding="Edema", state="present"),
            ),
            consequent=ClauseAtom(finding="Effusion", state="present"),
        ),
        date_introduced="2026-11-24",
    )

    rule2 = SymbolicRule(
        rule_id="R1.2",
        source_guideline="Fleischner",
        guideline_quote="Consolidation implies Infiltration.",
        formal_clause=FormalClause(
            antecedents=(ClauseAtom(finding="Consolidation", state="present"),),
            consequent=ClauseAtom(finding="Infiltration", state="present"),
        ),
        date_introduced="2026-11-24",
    )

    rules = (rule1, rule2)

    # Case 1: Normal study (all absent) -> both inactive, 0 violated, 0 supporting
    res_normal = evaluate_rules({}, rules)
    assert not res_normal.is_violated
    assert len(res_normal.supporting_rule_ids) == 0
    assert len(res_normal.violated_rule_ids) == 0

    # Case 2: R1.1 satisfied and supporting (Cardiomegaly=present, Edema=present, Effusion=present)
    findings_supporting = {
        "Cardiomegaly": "present",
        "Edema": "present",
        "Effusion": "present",
    }
    res_supp = evaluate_rules(findings_supporting, rules)
    assert not res_supp.is_violated
    assert res_supp.supporting_rule_ids == ("R1.1",)
    assert len(res_supp.violated_rule_ids) == 0

    # Case 3: R1.1 violated (Cardiomegaly=present, Edema=present, Effusion=absent)
    findings_violated = {
        "Cardiomegaly": "present",
        "Edema": "present",
        "Effusion": "absent",
    }
    res_viol = evaluate_rules(findings_violated, rules)
    assert res_viol.is_violated
    assert res_viol.violated_rule_ids == ("R1.1",)
    assert len(res_viol.supporting_rule_ids) == 0

    # Case 4: One supporting, one violated
    findings_mixed = {
        "Cardiomegaly": "present",
        "Edema": "present",
        "Effusion": "present",  # R1.1 satisfied & supporting
        "Consolidation": "present",
        "Infiltration": "absent",  # R1.2 satisfied & violated
    }
    res_mixed = evaluate_rules(findings_mixed, rules)
    assert res_mixed.is_violated
    assert res_mixed.supporting_rule_ids == ("R1.1",)
    assert res_mixed.violated_rule_ids == ("R1.2",)


def test_compute_normalized_entropy() -> None:
    """Verify normalized Shannon entropy boundaries and tensor shapes."""
    # 1. Deterministic one-hot distributions -> H = 0.0
    one_hot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    h_one_hot = compute_normalized_entropy(one_hot)
    np.testing.assert_allclose(h_one_hot, [0.0, 0.0, 0.0], atol=1e-6)

    # 2. Maximum uncertainty uniform distribution -> H = 1.0
    uniform = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
    h_uniform = compute_normalized_entropy(uniform)
    np.testing.assert_allclose(h_uniform, [1.0], atol=1e-6)

    # 3. Intermediate distribution
    intermediate = np.array([[0.7, 0.2, 0.1]])
    h_inter = compute_normalized_entropy(intermediate)
    assert 0.0 < float(h_inter[0]) < 1.0

    # 4. Multidimensional batch shape
    batch_probs = np.ones((5, 14, 3)) / 3.0
    h_batch = compute_normalized_entropy(batch_probs)
    assert h_batch.shape == (5, 14)
    np.testing.assert_allclose(h_batch, np.ones((5, 14)), atol=1e-6)

    # 5. Invalid shape raises ValueError
    with pytest.raises(ValueError, match="Expected last dimension to be 3"):
        compute_normalized_entropy(np.ones((2, 4)))
