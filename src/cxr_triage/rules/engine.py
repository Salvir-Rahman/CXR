"""Parameter-free, sub-millisecond Horn clause evaluation engine for symbolic clinical auditing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VALID_STATES: frozenset[str] = frozenset({"absent", "present", "severe"})
RULE_ID_PATTERN: re.Pattern[str] = re.compile(r"^R[12]\.\d+$")


@dataclass(frozen=True)
class ClauseAtom:
    """Atomic proposition representing finding and its discretized severity state."""

    finding: str
    state: str

    def __post_init__(self) -> None:
        if self.state not in VALID_STATES:
            raise ValueError(
                f"Invalid state '{self.state}'. State must be one of: {sorted(VALID_STATES)}"
            )


@dataclass(frozen=True)
class FormalClause:
    """Horn clause: conjunction of antecedents implying a single consequent."""

    antecedents: tuple[ClauseAtom, ...]
    consequent: ClauseAtom


@dataclass(frozen=True)
class SymbolicRule:
    """Auditable clinical rule with guideline provenance and formal Horn representation."""

    rule_id: str
    source_guideline: str
    guideline_quote: str
    formal_clause: FormalClause
    date_introduced: str


@dataclass(frozen=True)
class RuleEvaluationResult:
    """Result of evaluating a set of Horn clauses against observed findings."""

    supporting_rule_ids: tuple[str, ...]
    violated_rule_ids: tuple[str, ...]
    is_violated: bool
    total_rules_evaluated: int


def load_rule_registry(
    rules_path: Path, valid_labels: frozenset[str]
) -> tuple[tuple[SymbolicRule, ...], str]:
    """Load, validate, and hash the auditable clinical rule registry.

    Args:
        rules_path: Path to the JSON rule registry file.
        valid_labels: Set of canonical ontology labels permitted in rules.

    Returns:
        Tuple of (immutable tuple of validated SymbolicRule, SHA-256 hexadecimal digest).

    Raises:
        FileNotFoundError: If rules_path does not exist.
        TypeError: If rule objects or clause components have invalid types.
        ValueError: If JSON schema, fields, rule IDs, findings, or states are invalid.
    """
    if not rules_path.exists():
        raise FileNotFoundError(f"Rule registry file not found at: {rules_path}")

    raw_bytes = rules_path.read_bytes()
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    try:
        data: dict[str, Any] = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in rule registry: {e}") from e

    if "rules" not in data or not isinstance(data["rules"], list):
        raise ValueError("Rule registry JSON must contain a top-level 'rules' list.")

    parsed_rules: list[SymbolicRule] = []
    seen_rule_ids: set[str] = set()

    for idx, r_data in enumerate(data["rules"]):
        if not isinstance(r_data, dict):
            raise TypeError(f"Rule at index {idx} must be a dictionary.")

        # Validate 5 mandatory fields
        required_fields = {
            "rule_id",
            "source_guideline",
            "guideline_quote",
            "formal_clause",
            "date_introduced",
        }
        missing_fields = required_fields - set(r_data.keys())
        if missing_fields:
            raise ValueError(f"Rule at index {idx} is missing required fields: {missing_fields}")

        rule_id = str(r_data["rule_id"]).strip()
        if not RULE_ID_PATTERN.match(rule_id):
            raise ValueError(
                f"Rule ID '{rule_id}' at index {idx} does not match required pattern '^R[12]\\.\\d+$'."
            )
        if rule_id in seen_rule_ids:
            raise ValueError(f"Duplicate rule_id '{rule_id}' found in registry.")
        seen_rule_ids.add(rule_id)

        source_guideline = str(r_data["source_guideline"]).strip()
        guideline_quote = str(r_data["guideline_quote"]).strip()
        date_introduced = str(r_data["date_introduced"]).strip()

        if not source_guideline or not guideline_quote or not date_introduced:
            raise ValueError(f"Rule '{rule_id}' has empty textual provenance fields.")

        # Validate formal_clause
        clause_data = r_data["formal_clause"]
        if not isinstance(clause_data, dict):
            raise TypeError(f"Rule '{rule_id}' formal_clause must be a dict.")

        if "antecedents" not in clause_data or "consequent" not in clause_data:
            raise ValueError(
                f"Rule '{rule_id}' formal_clause must contain 'antecedents' and 'consequent'."
            )

        raw_antecedents = clause_data["antecedents"]
        raw_consequent = clause_data["consequent"]

        if not isinstance(raw_antecedents, list) or len(raw_antecedents) == 0:
            raise ValueError(f"Rule '{rule_id}' antecedents must be a non-empty list.")
        if not isinstance(raw_consequent, dict):
            raise TypeError(f"Rule '{rule_id}' consequent must be a dict.")

        parsed_antecedents: list[ClauseAtom] = []
        for a_idx, a in enumerate(raw_antecedents):
            if not isinstance(a, dict) or "finding" not in a or "state" not in a:
                raise ValueError(
                    f"Rule '{rule_id}' antecedent {a_idx} must have 'finding' and 'state'."
                )
            finding = str(a["finding"]).strip()
            state = str(a["state"]).strip().lower()
            if finding not in valid_labels:
                raise ValueError(
                    f"Rule '{rule_id}' contains out-of-ontology antecedent finding: '{finding}'."
                )
            parsed_antecedents.append(ClauseAtom(finding=finding, state=state))

        consequent_finding = str(raw_consequent.get("finding", "")).strip()
        consequent_state = str(raw_consequent.get("state", "")).strip().lower()

        if consequent_finding not in valid_labels:
            raise ValueError(
                f"Rule '{rule_id}' contains out-of-ontology consequent finding: '{consequent_finding}'."
            )

        parsed_consequent = ClauseAtom(finding=consequent_finding, state=consequent_state)

        formal_clause = FormalClause(
            antecedents=tuple(parsed_antecedents),
            consequent=parsed_consequent,
        )

        parsed_rules.append(
            SymbolicRule(
                rule_id=rule_id,
                source_guideline=source_guideline,
                guideline_quote=guideline_quote,
                formal_clause=formal_clause,
                date_introduced=date_introduced,
            )
        )

    return tuple(parsed_rules), sha256_hash


def evaluate_rules(
    discretized_findings: dict[str, str], rules: tuple[SymbolicRule, ...]
) -> RuleEvaluationResult:
    """Evaluate Horn clauses against observed discretized pathology findings.

    Horn clause semantics:
        - If all antecedents hold:
            - If consequent holds -> rule is SUPPORTING (antecedent satisfied & consequent satisfied)
            - If consequent does NOT hold -> rule is VIOLATED (antecedent satisfied & consequent violated)
        - If antecedents do NOT hold -> rule is inactive / vacuously satisfied.

    Args:
        discretized_findings: Mapping from finding name to discretized state ('absent', 'present', 'severe').
        rules: Tuple of validated SymbolicRule objects.

    Returns:
        RuleEvaluationResult containing supporting and violated rule IDs.
    """
    supporting: list[str] = []
    violated: list[str] = []

    for rule in rules:
        clause = rule.formal_clause

        # Check if all antecedents match
        antecedents_match = True
        for atom in clause.antecedents:
            observed_state = discretized_findings.get(atom.finding, "absent").lower()
            if observed_state != atom.state:
                antecedents_match = False
                break

        if not antecedents_match:
            # Rule inactive
            continue

        # Antecedents hold; check consequent
        consequent = clause.consequent
        observed_consequent_state = discretized_findings.get(
            consequent.finding, "absent"
        ).lower()

        if observed_consequent_state == consequent.state:
            supporting.append(rule.rule_id)
        else:
            violated.append(rule.rule_id)

    return RuleEvaluationResult(
        supporting_rule_ids=tuple(supporting),
        violated_rule_ids=tuple(violated),
        is_violated=len(violated) > 0,
        total_rules_evaluated=len(rules),
    )


def compute_normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy normalized by log(3) to [0, 1] for 3-state severity distributions.

    For a 3-class probability distribution p = [p_absent, p_present, p_severe]:
        H(p) = - sum_{i=1}^3 p_i * ln(p_i) / ln(3)

    Properties:
        - One-hot distribution [1, 0, 0] -> 0.0
        - Uniform distribution [1/3, 1/3, 1/3] -> 1.0

    Args:
        probs: Array of shape (..., 3) representing categorical probabilities.

    Returns:
        Array of shape (...) with normalized entropy values in [0, 1].
    """
    if probs.shape[-1] != 3:
        raise ValueError(
            f"Expected last dimension to be 3 (severity states), got shape {probs.shape}."
        )

    # Clip to avoid log(0) and clamp within probability simplex
    safe_probs = np.clip(probs, 1e-12, 1.0)
    # 0 * log(0) = 0 limit
    safe_probs = np.where(probs <= 0.0, 1.0, safe_probs)

    raw_entropy = -np.sum(np.where(probs > 0.0, probs * np.log(safe_probs), 0.0), axis=-1)
    normalized = raw_entropy / np.log(3.0)

    # Clamp numerically to exact [0.0, 1.0] range
    return np.clip(normalized, 0.0, 1.0)
