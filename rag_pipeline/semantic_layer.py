#!/usr/bin/env python3
"""
Brique 4.5 - Semantic Expansion Layer (V2)
===========================================
Enrichit les concepts trouves par le pipeline RAG (Briques 2-4) en exploitant
les relations semantiques de l'ontologie V2 (requires, supports, qualifiers,
excludes) AVANT le scoring.

Input:  found_ids = ["TACHYCARDIE", "QRS_LARGE", "MONOMORPHE", ...]
Output: SemanticResult avec patterns, findings, qualifiers, expanded patterns,
        et diagnostics implicites detectes par les findings seuls.

Dependances :
  - ontology_v2.json (produit par convert_owl_to_v2.py)

Auteur : BMad Team
Date   : 2026-03-31
"""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chargement ontologie V2 (singleton)
# ---------------------------------------------------------------------------

_ONTOLOGY_V2: Optional[Dict] = None


def _get_ontology_v2() -> Dict:
    """Charge l'ontologie V2 une seule fois."""
    global _ONTOLOGY_V2
    if _ONTOLOGY_V2 is None:
        candidates = [
            Path(__file__).parent.parent / "ECG lecture" / "data" / "ontology_v2.json",
            Path(__file__).parent / "data" / "ontology_v2.json",
        ]
        for p in candidates:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    _ONTOLOGY_V2 = json.load(f)
                n = len(_ONTOLOGY_V2.get("concepts", {}))
                logger.info(f"Ontologie V2 chargee : {p} ({n} concepts)")
                return _ONTOLOGY_V2
        raise FileNotFoundError(
            f"ontology_v2.json introuvable. Chemins testes : {[str(p) for p in candidates]}"
        )
    return _ONTOLOGY_V2


def load_ontology_v2(path) -> Dict:
    """Charge explicitement l'ontologie V2 depuis un chemin."""
    global _ONTOLOGY_V2
    with open(path, "r", encoding="utf-8") as f:
        _ONTOLOGY_V2 = json.load(f)
    logger.info(f"Ontologie V2 chargee : {path}")
    return _ONTOLOGY_V2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_key(key: str) -> str:
    """Normalise une cle en supprimant les accents (e->e, a->a, etc.).
    Pipeline V1 peut renvoyer des cles avec accents (ex: FAISCEAU_ACCESSOIRE_A_CONDUCTION_ANTEROGRADE)
    alors que les cles V2 sont sans accents."""
    nfkd = unicodedata.normalize("NFKD", key)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def get_concept(concept_id: str) -> Optional[Dict]:
    """Retourne le concept V2 ou None. Normalise les accents pour la correspondance."""
    onto = _get_ontology_v2()
    # Essai direct
    c = onto["concepts"].get(concept_id)
    if c:
        return c
    # Essai sans accents (pipeline V1 peut renvoyer des cles accentuees)
    normalized = normalize_key(concept_id)
    return onto["concepts"].get(normalized)


def get_concept_type(concept_id: str) -> str:
    """Retourne le type : pattern | finding | qualifier | topography."""
    c = get_concept(concept_id)
    return c["type"] if c else "unknown"


def is_hidden(concept_id: str) -> bool:
    """Retourne True si le concept a hide=1."""
    c = get_concept(concept_id)
    return bool(c and c.get("hide"))


def expand_qualifier_families(concept_id: str) -> List[str]:
    """Resout les qualifier families d'un concept en liste de qualifiers."""
    onto = _get_ontology_v2()
    c = get_concept(concept_id)
    if not c:
        return []
    families = c.get("has_qualifier_families", [])
    resolved = []
    for fam in families:
        fam_data = onto.get("qualifier_families", {}).get(fam, {})
        members = fam_data.get("members", []) if isinstance(fam_data, dict) else fam_data
        resolved.extend(members)
    return resolved


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatternExpansion:
    """Resultat de l'expansion semantique d'un pattern."""
    pattern_id: str
    requires: List[str] = field(default_factory=list)
    requires_satisfied: List[str] = field(default_factory=list)
    requires_missing: List[str] = field(default_factory=list)
    requires_ratio: float = 0.0
    supports_expected: List[str] = field(default_factory=list)
    supports_found: List[str] = field(default_factory=list)
    qualifiers_allowed: List[str] = field(default_factory=list)
    qualifiers_found: List[str] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)
    is_excluded: bool = False
    directly_found: bool = False


@dataclass
class ImplicitPattern:
    """Pattern non nomme par l'etudiant mais detectable par ses findings."""
    pattern_id: str
    requires: List[str] = field(default_factory=list)
    requires_satisfied: List[str] = field(default_factory=list)
    requires_ratio: float = 0.0
    supports_found: List[str] = field(default_factory=list)


@dataclass
class SemanticResult:
    """Resultat complet de l'expansion semantique."""
    # Classification des concepts trouves
    patterns: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    qualifiers: List[str] = field(default_factory=list)
    topography: List[str] = field(default_factory=list)
    hidden: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)

    # Expansion des patterns trouves
    expanded_patterns: Dict[str, PatternExpansion] = field(default_factory=dict)

    # Patterns implicites detectes par les findings
    implicit_patterns: Dict[str, ImplicitPattern] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialise le resultat pour le scoring."""
        return {
            "patterns": self.patterns,
            "findings": self.findings,
            "qualifiers": self.qualifiers,
            "topography": self.topography,
            "hidden": self.hidden,
            "unknown": self.unknown,
            "expanded_patterns": {
                pid: {
                    "requires": exp.requires,
                    "requires_satisfied": exp.requires_satisfied,
                    "requires_missing": exp.requires_missing,
                    "requires_ratio": round(exp.requires_ratio, 3),
                    "supports_found": exp.supports_found,
                    "qualifiers_found": exp.qualifiers_found,
                    "is_excluded": exp.is_excluded,
                    "directly_found": exp.directly_found,
                }
                for pid, exp in self.expanded_patterns.items()
            },
            "implicit_patterns": {
                pid: {
                    "requires": imp.requires,
                    "requires_satisfied": imp.requires_satisfied,
                    "requires_ratio": round(imp.requires_ratio, 3),
                    "supports_found": imp.supports_found,
                }
                for pid, imp in self.implicit_patterns.items()
            },
        }


# ---------------------------------------------------------------------------
# Core: Semantic Expansion
# ---------------------------------------------------------------------------

def expand_found_concepts(found_ids: List[str]) -> SemanticResult:
    """
    Point d'entree principal de la Brique 4.5.

    Prend la liste des ontology_ids trouves par les Briques 2-4 et :
    1. Classifie chaque concept (pattern/finding/qualifier/topography/hidden)
    2. Expand chaque pattern trouve (requires/supports/qualifiers/excludes)
    3. Cherche des patterns implicites via les findings orphelins

    Args:
        found_ids: Liste d'ontology_ids, ex: ["TACHYCARDIE", "QRS_LARGE", ...]

    Returns:
        SemanticResult complet pret pour le scoring V2
    """
    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    result = SemanticResult()

    # Normaliser les ids (pipeline V1 peut renvoyer des cles avec accents)
    found_ids = [normalize_key(fid) if fid not in concepts else fid for fid in found_ids]

    found_set = set(found_ids)

    # -------------------------------------------------------------------
    # Etape 1 : Classification contextuelle
    # -------------------------------------------------------------------
    # Un concept peut etre finding OU qualifier selon le contexte.
    # Ex: QRS_LARGE est "requires" de TV (= finding) mais "has_qualifiers"
    # d'un autre pattern (= qualifier). On resout par le contexte.
    #
    # Strategie :
    #  1. Identifier les patterns dans found_ids
    #  2. Pour chaque pattern, ses requires/supports => findings contextuels
    #  3. Pour chaque pattern, ses has_qualifiers => qualifiers contextuels
    #  4. Les concepts restants => type ontologique par defaut

    # Phase 1a : Identifier patterns et topography (sans ambiguite)
    remaining = []
    for cid in found_ids:
        if cid not in concepts:
            result.unknown.append(cid)
            continue
        if is_hidden(cid):
            result.hidden.append(cid)
            continue
        ctype = get_concept_type(cid)
        if ctype == "pattern":
            result.patterns.append(cid)
        elif ctype == "topography":
            result.topography.append(cid)
        else:
            remaining.append(cid)

    # Phase 1b : Construire les roles contextuels depuis les patterns trouves
    contextual_findings = set()
    contextual_qualifiers = set()
    for pid in result.patterns:
        c = get_concept(pid)
        if not c:
            continue
        for r in c.get("requires", []):
            contextual_findings.add(r)
        for s in c.get("supports", []):
            contextual_findings.add(s)
        for q in c.get("has_qualifiers", []):
            contextual_qualifiers.add(q)
        # Qualifier families -> expand members as contextual qualifiers
        for fam_q in expand_qualifier_families(pid):
            contextual_qualifiers.add(fam_q)

    # Phase 1c : Classer les concepts restants par contexte
    # Un concept peut etre a la fois finding ET qualifier (double role).
    # Ex: TACHYCARDIE est has_qualifiers de TSV (qualifier) ET requires de TV (finding).
    for cid in remaining:
        is_finding = cid in contextual_findings
        is_qualifier = cid in contextual_qualifiers

        if is_finding and is_qualifier:
            # Double role : present dans les deux listes
            result.findings.append(cid)
            result.qualifiers.append(cid)
        elif is_finding:
            result.findings.append(cid)
        elif is_qualifier:
            # Qualifier contextuel mais aussi disponible comme finding
            # pour satisfaire les requires d'autres patterns
            result.qualifiers.append(cid)
            result.findings.append(cid)
        else:
            # Pas lie a un pattern trouve -> type ontologique par defaut
            ctype = get_concept_type(cid)
            if ctype == "qualifier":
                # Un qualifier sans pattern = probablement un finding descriptif
                result.findings.append(cid)
            else:
                result.findings.append(cid)

    # -------------------------------------------------------------------
    # Etape 2 : Expand chaque pattern trouve
    # -------------------------------------------------------------------
    all_findings_and_patterns = set(result.findings) | set(result.patterns)
    all_qualifiers = set(result.qualifiers)

    for pid in result.patterns:
        exp = _expand_pattern(pid, all_findings_and_patterns, all_qualifiers, found_set)
        result.expanded_patterns[pid] = exp

    # -------------------------------------------------------------------
    # Etape 3 : Cherche patterns implicites via findings orphelins
    # -------------------------------------------------------------------
    # Un finding est "orphelin" s'il n'est requires d'aucun pattern trouve
    used_findings = set()
    for exp in result.expanded_patterns.values():
        used_findings.update(exp.requires_satisfied)

    orphan_findings = set(result.findings) - used_findings

    if orphan_findings:
        implicit = _find_implicit_patterns(
            orphan_findings, all_findings_and_patterns, all_qualifiers, found_set
        )
        result.implicit_patterns = implicit

    logger.info(
        f"Semantic expansion: {len(result.patterns)} patterns, "
        f"{len(result.findings)} findings, {len(result.qualifiers)} qualifiers, "
        f"{len(result.implicit_patterns)} implicit patterns detected"
    )

    return result


# ---------------------------------------------------------------------------
# Expansion d'un pattern
# ---------------------------------------------------------------------------

def _expand_pattern(
    pattern_id: str,
    all_findings: Set[str],
    all_qualifiers: Set[str],
    all_found: Set[str],
) -> PatternExpansion:
    """Expand un pattern : verifie requires, supports, qualifiers, excludes."""
    c = get_concept(pattern_id)
    if not c:
        return PatternExpansion(pattern_id=pattern_id)

    exp = PatternExpansion(pattern_id=pattern_id)
    exp.directly_found = pattern_id in all_found

    # --- requires ---
    exp.requires = c.get("requires", [])
    for r in exp.requires:
        if r in all_findings:
            exp.requires_satisfied.append(r)
        else:
            exp.requires_missing.append(r)
    if exp.requires:
        exp.requires_ratio = len(exp.requires_satisfied) / len(exp.requires)

    # --- supports ---
    exp.supports_expected = c.get("supports", [])
    exp.supports_found = [s for s in exp.supports_expected if s in all_findings]

    # --- qualifiers ---
    allowed_direct = set(c.get("has_qualifiers", []))
    allowed_families = set(expand_qualifier_families(pattern_id))
    exp.qualifiers_allowed = list(allowed_direct | allowed_families)
    exp.qualifiers_found = [
        q for q in all_qualifiers if q in allowed_direct or q in allowed_families
    ]

    # --- excludes ---
    excludes_direct = set(c.get("excludes", []))
    excludes_families_keys = c.get("excludes_families", [])
    onto = _get_ontology_v2()
    for fam_key in excludes_families_keys:
        fam_data = onto.get("qualifier_families", {}).get(fam_key, {})
        members = fam_data.get("members", []) if isinstance(fam_data, dict) else fam_data
        excludes_direct.update(members)
    # Also add children of excluded concepts
    for exc_id in list(excludes_direct):
        exc_c = get_concept(exc_id)
        if exc_c:
            excludes_direct.update(exc_c.get("children", []))
    exp.excludes = list(excludes_direct)
    exp.is_excluded = bool(excludes_direct & all_found)

    return exp


# ---------------------------------------------------------------------------
# Detection de patterns implicites
# ---------------------------------------------------------------------------

IMPLICIT_MIN_REQUIRES_RATIO = 1.0  # 100% des requires doivent etre satisfaits

def _find_implicit_patterns(
    orphan_findings: Set[str],
    all_findings: Set[str],
    all_qualifiers: Set[str],
    all_found: Set[str],
) -> Dict[str, ImplicitPattern]:
    """
    Cherche des patterns non nommes par l'etudiant mais dont les findings
    sont presents. C'est le coeur de la V2 : detecter que l'etudiant decrit
    un syndrome sans le nommer.

    Exemple : l'etudiant dit "QRS larges + tachycardie" -> on detecte
    TACHYCARDIE_VENTRICULAIRE comme pattern implicite.
    """
    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    implicit = {}

    for cid, c in concepts.items():
        if c.get("type") != "pattern":
            continue
        if cid in all_found:
            continue  # Deja trouve explicitement
        if is_hidden(cid):
            continue

        requires = c.get("requires", [])
        if not requires:
            continue

        satisfied = [r for r in requires if r in all_findings]
        ratio = len(satisfied) / len(requires)

        if ratio >= IMPLICIT_MIN_REQUIRES_RATIO:
            imp = ImplicitPattern(
                pattern_id=cid,
                requires=requires,
                requires_satisfied=satisfied,
                requires_ratio=ratio,
            )
            # Bonus : check supports
            supports = c.get("supports", [])
            imp.supports_found = [s for s in supports if s in all_findings]

            # Check exclusions - skip si un concept exclus est trouve
            excludes_direct = set(c.get("excludes", []))
            if excludes_direct & all_found:
                continue

            implicit[cid] = imp

    # Trier par ratio decroissant
    implicit = dict(
        sorted(implicit.items(), key=lambda x: x[1].requires_ratio, reverse=True)
    )

    return implicit


# ---------------------------------------------------------------------------
# Utilitaire : resume lisible
# ---------------------------------------------------------------------------

def format_semantic_summary(result: SemanticResult) -> str:
    """Produit un resume lisible de l'expansion semantique."""
    lines = []
    lines.append("=== Semantic Expansion Summary ===")
    lines.append(f"Patterns trouves:   {len(result.patterns)} -> {result.patterns}")
    lines.append(f"Findings trouves:   {len(result.findings)} -> {result.findings}")
    lines.append(f"Qualifiers trouves: {len(result.qualifiers)} -> {result.qualifiers}")
    lines.append(f"Topographie:        {len(result.topography)} -> {result.topography}")
    if result.hidden:
        lines.append(f"Hidden (ignores):   {result.hidden}")
    if result.unknown:
        lines.append(f"Inconnus:           {result.unknown}")

    lines.append("")
    for pid, exp in result.expanded_patterns.items():
        lines.append(f"--- Pattern: {pid} ---")
        lines.append(f"  Trouve directement: {exp.directly_found}")
        if exp.requires:
            lines.append(f"  Requires:    {exp.requires}")
            lines.append(f"    Satisfaits:  {exp.requires_satisfied}")
            lines.append(f"    Manquants:   {exp.requires_missing}")
            lines.append(f"    Ratio:       {exp.requires_ratio:.0%}")
        if exp.supports_found:
            lines.append(f"  Supports trouves: {exp.supports_found}")
        if exp.qualifiers_found:
            lines.append(f"  Qualifiers trouves: {exp.qualifiers_found}")
        if exp.is_excluded:
            lines.append(f"  EXCLU par: {exp.excludes}")

    if result.implicit_patterns:
        lines.append("")
        lines.append("=== Patterns IMPLICITES detectes ===")
        for pid, imp in result.implicit_patterns.items():
            lines.append(f"  {pid}: ratio={imp.requires_ratio:.0%}")
            lines.append(f"    requires={imp.requires}")
            lines.append(f"    satisfaits={imp.requires_satisfied}")
            if imp.supports_found:
                lines.append(f"    supports trouves={imp.supports_found}")

    return "\n".join(lines)
