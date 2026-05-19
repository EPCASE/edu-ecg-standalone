#!/usr/bin/env python3
"""
Brique 5 - Scoring V2 Composite
=================================
Scoring composite exploitant l'expansion semantique de la Brique 4.5.

Score par pattern = base (50) + requires (30) + qualifiers (10) + supports (5)
                  - exclusion (= 0)

Auteur : BMad Team
Date   : 2026-03-31
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from semantic_layer import (
    SemanticResult,
    PatternExpansion,
    ImplicitPattern,
    expand_found_concepts,
    get_concept,
    is_hidden,
    load_ontology_v2,
    normalize_key,
    _get_ontology_v2,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hierarchy helpers (parent ↔ child navigation)
# ---------------------------------------------------------------------------

def get_all_ancestors(concept_id: str, max_depth: int = 5) -> Dict[str, int]:
    """Return {ancestor_id: distance} for a concept. Distance 1 = direct parent."""
    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    result: Dict[str, int] = {}

    def _walk(cid: str, depth: int):
        if depth > max_depth:
            return
        c = concepts.get(cid)
        if not c:
            return
        for pid in c.get("parents", []):
            if pid not in result or depth < result[pid]:
                result[pid] = depth
                _walk(pid, depth + 1)

    _walk(concept_id, 1)
    return result


def get_all_descendants(concept_id: str, max_depth: int = 5) -> Dict[str, int]:
    """Return {descendant_id: distance} for a concept. Distance 1 = direct child."""
    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    result: Dict[str, int] = {}

    def _walk(cid: str, depth: int):
        if depth > max_depth:
            return
        c = concepts.get(cid)
        if not c:
            return
        for chid in c.get("children", []):
            if chid not in result or depth < result[chid]:
                result[chid] = depth
                _walk(chid, depth + 1)

    _walk(concept_id, 1)
    return result


def hierarchical_match(found_id: str, expected_id: str) -> Optional[Dict]:
    """Check if found_id is a parent or child of expected_id.
    Returns dict with match_type and distance, or None."""
    if found_id == expected_id:
        return {"match_type": "exact", "distance": 0}

    # Is found a parent of expected? (student less specific)
    ancestors_of_expected = get_all_ancestors(expected_id)
    if found_id in ancestors_of_expected:
        return {"match_type": "parent", "distance": ancestors_of_expected[found_id]}

    # Is found a child of expected? (student more specific)
    descendants_of_expected = get_all_descendants(expected_id)
    if found_id in descendants_of_expected:
        return {"match_type": "child", "distance": descendants_of_expected[found_id]}

    # Are they siblings? (same parent)
    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    parents_found = set(concepts.get(found_id, {}).get("parents", []))
    parents_expected = set(concepts.get(expected_id, {}).get("parents", []))
    common_parents = parents_found & parents_expected
    if common_parents:
        return {"match_type": "sibling", "distance": 2, "common_parent": sorted(common_parents)[0]}

    return None


def supports_match(found_id: str, expected_id: str) -> Optional[Dict]:
    """Check if found_id appears in the requires/supports/has_qualifiers
    of expected_id or any of its ancestors (walking up the tree).
    
    Use case: golden=FA, student says "tachycardie" → TACHYCARDIE is in
    has_qualifiers of TSV (parent of FA) → match_type="supports".
    """
    onto = _get_ontology_v2()
    concepts = onto["concepts"]

    # Collect expected + all its ancestors
    targets = {expected_id: 0}
    targets.update(get_all_ancestors(expected_id))

    # Check requires, has_qualifiers, supports (in priority order)
    RELATIONS = ["requires", "has_qualifiers", "supports"]

    for target_id, distance in targets.items():
        c = concepts.get(target_id, {})
        for rel in RELATIONS:
            if found_id in c.get(rel, []):
                return {
                    "match_type": "supports",
                    "relation": rel,
                    "via": target_id,
                    "distance": distance,
                }

    return None


# ---------------------------------------------------------------------------
# Scoring weights (from ontology_v2.json scoring_rules)
# ---------------------------------------------------------------------------

PATTERN_BASE_SCORE = 50
REQUIRES_WEIGHT = 30
QUALIFIER_WEIGHT = 10
SUPPORTS_WEIGHT = 5

CATEGORIE_MULTIPLIER = {
    "DIAGNOSTIC_URGENT": 5,
    "DIAGNOSTIC_MAJEUR": 4,
    "DIAGNOSTIC_MOYEN": 3,
    "DESCRIPTION_ECG": 2,
    "QUALIFICATEUR": 1,
    "TOPOGRAPHIE": 1,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatternScore:
    """Score detaille d'un pattern."""
    pattern_id: str
    concept_name: str = ""
    categorie: str = ""
    poids: int = 2

    base_score: float = 0.0
    requires_score: float = 0.0
    qualifier_score: float = 0.0
    supports_score: float = 0.0
    exclusion_penalty: bool = False

    raw_score: float = 0.0
    weighted_score: float = 0.0

    requires_ratio: float = 0.0
    requires_satisfied: List[str] = field(default_factory=list)
    requires_missing: List[str] = field(default_factory=list)
    qualifiers_found: List[str] = field(default_factory=list)
    supports_found: List[str] = field(default_factory=list)
    is_implicit: bool = False
    directly_found: bool = False

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "concept_name": self.concept_name,
            "categorie": self.categorie,
            "poids": self.poids,
            "base_score": self.base_score,
            "requires_score": round(self.requires_score, 1),
            "qualifier_score": self.qualifier_score,
            "supports_score": self.supports_score,
            "exclusion_penalty": self.exclusion_penalty,
            "raw_score": round(self.raw_score, 1),
            "weighted_score": round(self.weighted_score, 1),
            "requires_ratio": round(self.requires_ratio, 3),
            "requires_satisfied": self.requires_satisfied,
            "requires_missing": self.requires_missing,
            "qualifiers_found": self.qualifiers_found,
            "supports_found": self.supports_found,
            "is_implicit": self.is_implicit,
            "directly_found": self.directly_found,
        }


@dataclass
class ScoringResult:
    """Resultat complet du scoring V2."""
    pattern_scores: List[PatternScore] = field(default_factory=list)
    finding_scores: List[Dict] = field(default_factory=list)
    total_score: float = 0.0
    max_possible_score: float = 0.0
    score_ratio: float = 0.0
    semantic_result: Optional[SemanticResult] = None

    def to_dict(self) -> Dict:
        return {
            "pattern_scores": [ps.to_dict() for ps in self.pattern_scores],
            "finding_scores": self.finding_scores,
            "total_score": round(self.total_score, 1),
            "max_possible_score": round(self.max_possible_score, 1),
            "score_ratio": round(self.score_ratio, 3),
            "summary": {
                "patterns_found": sum(1 for ps in self.pattern_scores if ps.directly_found),
                "patterns_implicit": sum(1 for ps in self.pattern_scores if ps.is_implicit),
                "patterns_excluded": sum(1 for ps in self.pattern_scores if ps.exclusion_penalty),
                "findings_matched": len(self.finding_scores),
            },
        }


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_pattern(expansion: PatternExpansion, is_implicit: bool = False) -> PatternScore:
    """
    Score composite d'un pattern.
    Score = base (50 si trouve) + requires (30 * ratio) + qualifiers (10 * n, cap 30)
          + supports (5 * n, cap 15) - exclusion (= 0)
    """
    c = get_concept(expansion.pattern_id)
    ps = PatternScore(pattern_id=expansion.pattern_id)

    if c:
        ps.concept_name = c.get("concept_name", expansion.pattern_id)
        ps.categorie = c.get("categorie", "DESCRIPTION_ECG")
        ps.poids = c.get("poids", 2)

    ps.is_implicit = is_implicit
    ps.directly_found = expansion.directly_found

    if expansion.is_excluded:
        ps.exclusion_penalty = True
        ps.raw_score = 0.0
        ps.weighted_score = 0.0
        return ps

    if expansion.directly_found:
        ps.base_score = PATTERN_BASE_SCORE

    ps.requires_ratio = expansion.requires_ratio
    ps.requires_satisfied = expansion.requires_satisfied
    ps.requires_missing = expansion.requires_missing
    ps.requires_score = REQUIRES_WEIGHT * expansion.requires_ratio

    ps.qualifiers_found = expansion.qualifiers_found
    ps.qualifier_score = min(QUALIFIER_WEIGHT * len(expansion.qualifiers_found), 30)

    ps.supports_found = expansion.supports_found
    ps.supports_score = min(SUPPORTS_WEIGHT * len(expansion.supports_found), 15)

    ps.raw_score = ps.base_score + ps.requires_score + ps.qualifier_score + ps.supports_score

    multiplier = CATEGORIE_MULTIPLIER.get(ps.categorie, 2)
    ps.weighted_score = ps.raw_score * (multiplier / 4.0)

    return ps


def score_finding(concept_id: str) -> Dict:
    """Score simple pour un finding orphelin."""
    c = get_concept(concept_id)
    if not c:
        return {"concept_id": concept_id, "score": 1, "poids": 1}
    poids = c.get("poids", 2)
    return {
        "concept_id": concept_id,
        "concept_name": c.get("concept_name", concept_id),
        "categorie": c.get("categorie", "DESCRIPTION_ECG"),
        "poids": poids,
        "score": poids,
    }


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def _match_found_to_expected(
    found_id: str,
    expected_id: str,
    sem: SemanticResult,
) -> Optional[Dict]:
    """
    Determine si un concept trouve (found_id) correspond a un concept
    attendu (expected_id) du golden set.

    Retourne un dict {match_type, score_factor} ou None.
    score_factor : 1.0 = exact, 0.8 = child, 0.6 = parent, 0.5 = sibling,
                   0.7 = implicit, 0.5 = supports
    """
    nf = normalize_key(found_id)
    ne = normalize_key(expected_id)

    # 1. Exact match
    if nf == ne:
        return {"match_type": "exact", "score_factor": 1.0}

    # 2. Implicit pattern match (found_ids → implicit patterns contain expected)
    if ne in sem.implicit_patterns:
        return {"match_type": "implicit", "score_factor": 0.7}

    # 3. Hierarchical match (parent/child/sibling)
    hm = hierarchical_match(nf, ne)
    if hm:
        factors = {"child": 0.8, "parent": 0.6, "sibling": 0.5}
        return {
            "match_type": hm["match_type"],
            "score_factor": factors.get(hm["match_type"], 0.5),
            "distance": hm["distance"],
        }

    # 4. Supports match (found is in requires/supports of expected)
    sm = supports_match(nf, ne)
    if sm:
        return {
            "match_type": "supports",
            "score_factor": 0.5,
            "relation": sm["relation"],
        }

    return None


def score_student_response_v2(
    found_ids: List[str],
    expected_ids: Optional[List[str]] = None,
) -> ScoringResult:
    """
    Point d'entree principal du scoring V2.

    Scoring centre sur le golden set (expected_ids) :
    1. Appelle la Brique 4.5 (expansion semantique) sur les found_ids
    2. Pour chaque expected_id, cherche le meilleur match parmi found_ids
    3. Score = poids_ontologie * score_factor pour chaque expected matche
    4. Le score ne peut pas depasser max_possible_score (somme des poids)
    """
    sem = expand_found_concepts(found_ids)
    result = ScoringResult()
    result.semantic_result = sem

    if not expected_ids:
        # Sans golden set, pas de scoring possible
        return result

    # Normaliser
    found_ids_norm = [normalize_key(fid) for fid in found_ids]
    expected_ids_norm = [normalize_key(eid) for eid in expected_ids]

    # Pour chaque expected, trouver le meilleur match parmi les found
    used_found = set()
    matched_expected: List[Dict] = []

    # Priorite : exact > implicit > child > parent > sibling > supports
    match_priority = {"exact": 0, "implicit": 1, "child": 2, "parent": 3,
                      "sibling": 4, "supports": 5}

    # Collecter tous les candidats (expected_id, found_id, match_info)
    all_candidates = []
    for eid in expected_ids_norm:
        # Check implicit patterns first (no found_id needed)
        if eid in sem.implicit_patterns:
            all_candidates.append({
                "expected_id": eid,
                "found_id": None,
                "match_type": "implicit",
                "score_factor": 0.7,
            })
        # Check each found_id
        for fid in found_ids_norm:
            m = _match_found_to_expected(fid, eid, sem)
            if m:
                all_candidates.append({
                    "expected_id": eid,
                    "found_id": fid,
                    **m,
                })

    # Sort: best match_type first, then highest score_factor
    all_candidates.sort(
        key=lambda x: (match_priority.get(x["match_type"], 9), -x["score_factor"])
    )

    # Greedy assignment: each expected and found used at most once
    matched_expected_set = set()
    for cand in all_candidates:
        eid = cand["expected_id"]
        fid = cand["found_id"]
        if eid in matched_expected_set:
            continue
        if fid is not None and fid in used_found:
            continue
        matched_expected.append(cand)
        matched_expected_set.add(eid)
        if fid is not None:
            used_found.add(fid)

    # Score each matched expected_id using ontology poids
    for match in matched_expected:
        eid = match["expected_id"]
        factor = match["score_factor"]

        c = get_concept(eid)
        poids = c.get("poids", 2) if c else 2
        concept_name = c.get("concept_name", eid) if c else eid
        categorie = c.get("categorie", "DESCRIPTION_ECG") if c else "DESCRIPTION_ECG"

        score = poids * factor

        result.finding_scores.append({
            "concept_id": eid,
            "concept_name": concept_name,
            "categorie": categorie,
            "poids": poids,
            "score": round(score, 2),
            "match_type": match["match_type"],
            "score_factor": factor,
            "matched_via": match.get("found_id", ""),
        })

    # Totals (scoring unifie sur poids)
    result.total_score = sum(fs.get("score", 0) for fs in result.finding_scores)

    result.max_possible_score = _calculate_max_score(expected_ids_norm)
    if result.max_possible_score > 0:
        result.score_ratio = min(result.total_score / result.max_possible_score, 1.0)

    return result


def _calculate_max_score(expected_ids: List[str]) -> float:
    """Score maximum possible = somme des poids des concepts attendus."""
    total = 0.0
    for eid in expected_ids:
        c = get_concept(eid)
        if not c:
            total += 2  # poids par defaut
            continue
        total += c.get("poids", 2)
    return total


# ---------------------------------------------------------------------------
# Comparison with golden set
# ---------------------------------------------------------------------------

def compare_with_golden_set(
    found_ids: List[str],
    expected_ids: List[str],
) -> Dict:
    """Compare la reponse etudiant avec le golden set expert.
    
    Includes hierarchical matching: if a student finds a parent/child/sibling
    of an expected concept, it counts as a partial match.
    """
    # Normaliser les cles (accents pipeline V1 vs cles V2 sans accents)
    found_ids = [normalize_key(fid) for fid in found_ids]
    expected_ids = [normalize_key(eid) for eid in expected_ids]

    found_set = set(found_ids)
    expected_set = set(expected_ids)

    # --- Exact matches ---
    matched_exact = found_set & expected_set
    remaining_expected = expected_set - matched_exact
    remaining_found = found_set - matched_exact

    # --- Implicit matches (via semantic expansion) ---
    student_score = score_student_response_v2(found_ids, expected_ids)
    implicit_matches = set()
    if student_score.semantic_result:
        for pid in student_score.semantic_result.implicit_patterns:
            if pid in remaining_expected:
                implicit_matches.add(pid)
    remaining_expected -= implicit_matches

    # --- Hierarchical matches (parent/child/sibling) ---
    matched_hierarchical = []  # list of {found, expected, match_type, distance}
    hier_found_used = set()
    hier_expected_used = set()

    # Try to match remaining expected with remaining found via hierarchy
    # Prioritize closer matches (distance=1 first)
    candidates = []
    for eid in remaining_expected:
        for fid in remaining_found:
            hm = hierarchical_match(fid, eid)
            if hm is not None:
                candidates.append({
                    "found": fid,
                    "expected": eid,
                    **hm,
                })
    # Sort: prefer exact type matches, then by distance
    type_priority = {"child": 0, "parent": 1, "sibling": 2}
    candidates.sort(key=lambda x: (type_priority.get(x["match_type"], 9), x["distance"]))

    for c in candidates:
        if c["found"] in hier_found_used or c["expected"] in hier_expected_used:
            continue
        matched_hierarchical.append(c)
        hier_found_used.add(c["found"])
        hier_expected_used.add(c["expected"])

    remaining_expected -= hier_expected_used
    remaining_found -= hier_found_used

    # --- Supports matches (found is in requires/supports of expected or ancestor) ---
    matched_supports = []
    sup_found_used = set()
    sup_expected_used = set()

    sup_candidates = []
    for eid in remaining_expected:
        for fid in remaining_found:
            sm = supports_match(fid, eid)
            if sm is not None:
                sup_candidates.append({"found": fid, "expected": eid, **sm})
    # Prefer requires over supports, then closer distance
    rel_priority = {"requires": 0, "supports": 1}
    sup_candidates.sort(key=lambda x: (rel_priority.get(x["relation"], 9), x["distance"]))

    for c in sup_candidates:
        if c["found"] in sup_found_used or c["expected"] in sup_expected_used:
            continue
        matched_supports.append(c)
        sup_found_used.add(c["found"])
        sup_expected_used.add(c["expected"])

    remaining_expected -= sup_expected_used
    remaining_found -= sup_found_used

    return {
        "matched_exact": sorted(matched_exact),
        "matched_implicit": sorted(implicit_matches),
        "matched_hierarchical": matched_hierarchical,
        "matched_supports": matched_supports,
        "missed": sorted(remaining_expected),
        "extra": sorted(remaining_found),
        "student_score": student_score.to_dict(),
        "counts": {
            "expected": len(expected_ids),
            "found": len(found_ids),
            "matched_exact": len(matched_exact),
            "matched_implicit": len(implicit_matches),
            "matched_hierarchical": len(matched_hierarchical),
            "matched_supports": len(matched_supports),
            "missed": len(remaining_expected),
            "extra": len(remaining_found),
        },
    }


# ---------------------------------------------------------------------------
# Format lisible
# ---------------------------------------------------------------------------

def format_scoring_summary(result: ScoringResult) -> str:
    """Resume lisible du scoring."""
    lines = []
    lines.append("=== Scoring V2 Summary ===")
    lines.append(f"Total score: {result.total_score:.1f}")
    if result.max_possible_score > 0:
        lines.append(f"Max possible: {result.max_possible_score:.1f}")
        lines.append(f"Ratio: {result.score_ratio:.0%}")

    if result.pattern_scores:
        lines.append("")
        lines.append("--- Pattern Scores ---")
        for ps in result.pattern_scores:
            tag = "[IMPLICIT]" if ps.is_implicit else "[EXPLICIT]"
            excl = " [EXCLUDED]" if ps.exclusion_penalty else ""
            lines.append(
                f"  {tag} {ps.pattern_id}: "
                f"raw={ps.raw_score:.1f} weighted={ps.weighted_score:.1f}{excl}"
            )
            if ps.requires_missing:
                lines.append(f"    requires manquants: {ps.requires_missing}")

    if result.finding_scores:
        lines.append("")
        lines.append("--- Golden Set Matching ---")
        for fs in result.finding_scores:
            mt = fs.get("match_type", "?")
            via = fs.get("matched_via", "")
            via_str = f" (via {via})" if via and via != fs["concept_id"] else ""
            lines.append(
                f"  {fs['concept_id']}: score={fs['score']}/{fs['poids']} "
                f"[{mt}{via_str}]"
            )

    return "\n".join(lines)
