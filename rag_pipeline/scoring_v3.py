#!/usr/bin/env python3
"""
Brique 5 — Scoring V3 Ontologique
====================================
Scoring centré sur les concepts du golden set, exploitant directement
les relations ontologiques (requires, has_qualifiers, supports, excludes).

Principe :
    Score = Σ score(concept_i) / N concepts golden

    Pour chaque concept attendu :
    1.  Trouvé exact (ou synonyme) dans found_ids         → 1.0
    1b. Un enfant (descendant) trouvé dans found_ids       → 1.0
        (plus spécifique que l'attendu → mérite tous les points)
    1c. Un parent trouvé dans found_ids :
        - parent direct (distance 1)                       → 2/3
        - parent éloigné (distance ≥ 2)                    → 1/3
        Ce score sert de plancher ; on continue à vérifier
        requires/qualifiers/supports et on garde le MAX.
    2.  Non trouvé, mais a requires → nb_req_trouvés / nb_req   (0..1)
       - Si un require n'est pas trouvé, on vérifie récursivement ses
         propres requires/qualifiers/supports (profondeur max 2).
         Ex: QRS_FINS trouvé → QRS_NORMAL reçoit 0.5 (1/2 sub-requires).
    3.  Non trouvé, pas de requires, has_qualifiers trouvés → 2/3
    4.  Non trouvé, pas de requires, supports trouvés       → 1/3
    5.  Un excludes ou excludes_families trouvé             → 0 (écrase tout)
    6.  Rien                                               → 0

    On ne cumule PAS les sources : on prend toujours le score max
    (enfant > parent+1 > qualifier > parent+2 > support).

Conversion des négations :
    Les concepts avec statut "absent" sont convertis en concepts positifs
    via le mapping ontologique (excludes / excludes_families).
    Ex: absent(TROUBLE_DE_REPOLARISATION) → PAS_D_ANOMALIE_DE_LE_REPOLARISATION
    Si pas de mapping → ignoré.

Auteur : BMad Team
Date   : 2026-04-06
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from semantic_layer import (
    get_concept,
    load_ontology_v2,
    normalize_key,
    _get_ontology_v2,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConceptScore:
    """Score détaillé d'un concept golden."""
    concept_id: str
    concept_name: str = ""
    score: float = 0.0
    max_score: float = 1.0
    match_type: str = "missed"  # exact | requires | qualifier | support | excluded | missed
    detail: str = ""
    # Pour requires
    requires_total: int = 0
    requires_found: int = 0
    requires_satisfied: List[str] = field(default_factory=list)
    requires_missing: List[str] = field(default_factory=list)
    # Pour qualifier/support
    qualifiers_found: List[str] = field(default_factory=list)
    supports_found: List[str] = field(default_factory=list)
    # Pour exclusion
    excluded_by: str = ""


# ---------------------------------------------------------------------------
# Lookup de concepts (V2-compatible, remplace scoring_v1.find_owl_concept)
# ---------------------------------------------------------------------------

def find_owl_concept(concept_text: str) -> Optional[Dict]:
    """
    Cherche un concept dans l'ontologie V2 par son label français.

    Stratégie de recherche (par ordre de priorité) :
      1. Match exact sur concept_name (case-insensitive)
      2. Match exact sur un synonyme
      3. Match partiel (contient / est contenu dans)

    Args:
        concept_text: Le texte du concept (ex: "Fibrillation atriale", "BBG")

    Returns:
        dict avec ontology_id, concept_name, poids, categorie, synonymes.
        Retourne un dict par défaut (poids=1) si non trouvé.
    """
    onto = _get_ontology_v2()
    concepts = onto.get("concepts", {})
    concept_lower = concept_text.lower().strip()

    # 1. Recherche exacte par concept_name
    for cid, cdata in concepts.items():
        if cdata.get("concept_name", "").lower().strip('"') == concept_lower:
            return {
                "ontology_id": cid,
                "concept_name": cdata.get("concept_name", "").strip('"'),
                "poids": cdata.get("poids", 1),
                "categorie": cdata.get("categorie", "DESCRIPTEUR_ECG"),
                "synonymes": cdata.get("synonymes", []),
            }

    # 2. Recherche par synonymes
    for cid, cdata in concepts.items():
        synonymes = [s.lower() for s in cdata.get("synonymes", [])]
        if concept_lower in synonymes:
            return {
                "ontology_id": cid,
                "concept_name": cdata.get("concept_name", "").strip('"'),
                "poids": cdata.get("poids", 1),
                "categorie": cdata.get("categorie", "DESCRIPTEUR_ECG"),
                "synonymes": cdata.get("synonymes", []),
            }

    # 3. Recherche partielle (contient)
    for cid, cdata in concepts.items():
        cname = cdata.get("concept_name", "").lower().strip('"')
        if concept_lower in cname or cname in concept_lower:
            return {
                "ontology_id": cid,
                "concept_name": cdata.get("concept_name", "").strip('"'),
                "poids": cdata.get("poids", 1),
                "categorie": cdata.get("categorie", "DESCRIPTEUR_ECG"),
                "synonymes": cdata.get("synonymes", []),
            }

    # Pas trouvé → dict par défaut
    return {
        "ontology_id": concept_text.upper().replace(" ", "_"),
        "concept_name": concept_text,
        "poids": 1,
        "categorie": "DESCRIPTEUR_ECG",
        "synonymes": [],
    }


@dataclass
class ScoringResultV3:
    """Résultat complet du scoring V3."""
    concept_scores: List[ConceptScore] = field(default_factory=list)
    total_score: float = 0.0
    max_possible_score: float = 0.0
    score_pct: float = 0.0
    n_exact: int = 0
    n_requires: int = 0
    n_qualifier: int = 0
    n_support: int = 0
    n_excluded: int = 0
    n_missed: int = 0
    # Négations converties
    negation_conversions: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Negation mapping  (absent → positive concept)
# ---------------------------------------------------------------------------

_NEGATION_MAP: Optional[Dict[str, str]] = None


def _is_normal_concept(concept_id: str) -> bool:
    """Heuristique : le concept représente la normalité (nom contient normal/pas d'/absence d')."""
    onto = _get_ontology_v2()
    c = onto["concepts"].get(concept_id, {})
    name = c.get("concept_name", "").lower()
    syns = [s.lower() for s in c.get("synonymes", [])]
    return any(
        "normal" in t or "pas d" in t or "absence d" in t or "physiolog" in t
        for t in [name] + syns
    )


def build_negation_map() -> Dict[str, str]:
    """
    Construit le mapping absent(patho) → concept de normalité
    depuis les excludes / excludes_families de l'ontologie V2.

    Priorité : excludes directs (spécifiques) > excludes_families (génériques).
    """
    global _NEGATION_MAP
    if _NEGATION_MAP is not None:
        return _NEGATION_MAP

    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    mapping: Dict[str, str] = {}

    # Pass 1 : excludes directs (haute priorité, plus spécifiques)
    for nid, nc in concepts.items():
        excl = nc.get("excludes", [])
        if not excl or not _is_normal_concept(nid):
            continue
        for x in excl:
            mapping[x] = nid

    # Pass 2 : excludes_families (basse priorité, ne remplace pas)
    for nid, nc in concepts.items():
        excl_fam = nc.get("excludes_families", [])
        if not excl_fam or not _is_normal_concept(nid):
            continue
        for fam in excl_fam:
            if fam not in mapping:
                mapping[fam] = nid
            for child in _get_all_children_recursive(fam, max_depth=3):
                if child not in mapping:
                    mapping[child] = nid

    _NEGATION_MAP = mapping
    logger.info("Negation map: %d entries", len(mapping))
    return mapping


def convert_absents_to_positive(absent_ids: List[str]) -> List[Tuple[str, str]]:
    """
    Convertit les concepts absents en concepts positifs.

    Args:
        absent_ids: IDs des concepts avec statut "absent" dans le NER

    Returns:
        Liste de tuples (absent_id, positive_id) pour les conversions réussies.
        Les absents sans mapping sont ignorés silencieusement.
    """
    neg_map = build_negation_map()
    conversions: List[Tuple[str, str]] = []
    for aid in absent_ids:
        naid = normalize_key(aid)
        if naid in neg_map:
            conversions.append((naid, neg_map[naid]))
    return conversions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_all_children_recursive(concept_id: str, max_depth: int = 3) -> Set[str]:
    """Retourne tous les enfants (descendants) d'un concept."""
    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    result = set()

    def _walk(cid, depth):
        if depth > max_depth:
            return
        c = concepts.get(cid, {})
        for child in c.get("children", []):
            if child not in result:
                result.add(child)
                _walk(child, depth + 1)

    _walk(concept_id, 0)
    return result


def _check_excludes(
    concept_id: str,
    found_set: Set[str],
) -> Optional[str]:
    """
    Vérifie si un concept trouvé par l'étudiant est dans les excludes
    ou excludes_families du concept attendu.
    Retourne l'ID du concept excluant trouvé, ou None.
    """
    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    c = concepts.get(concept_id, {})

    # Excludes directs
    for exc in c.get("excludes", []):
        if exc in found_set:
            return exc

    # Excludes families : chaque famille = un concept parent dont tous
    # les descendants sont exclus
    for fam in c.get("excludes_families", []):
        if fam in found_set:
            return fam
        children = _get_all_children_recursive(fam)
        hit = found_set & children
        if hit:
            return sorted(hit)[0]

    return None


def _get_all_parents_recursive(concept_id: str, max_depth: int = 3) -> Dict[str, int]:
    """
    Retourne tous les ancêtres d'un concept avec leur distance.
    {parent_id: distance} où distance = 1 pour parent direct, 2 pour grand-parent, etc.
    """
    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    result: Dict[str, int] = {}

    def _walk(cid: str, depth: int):
        if depth > max_depth:
            return
        c = concepts.get(cid, {})
        for parent in c.get("parents", []):
            if parent not in result or depth < result[parent]:
                result[parent] = depth
                _walk(parent, depth + 1)

    _walk(concept_id, 1)
    return result


def _find_child_in_found(concept_id: str, found_set: Set[str]) -> Optional[str]:
    """
    Vérifie si un enfant (descendant) du concept attendu est dans found_set.
    Un enfant est plus spécifique → mérite les points complets.
    Retourne le premier enfant trouvé, ou None.
    """
    children = _get_all_children_recursive(concept_id, max_depth=3)
    hit = found_set & children
    return sorted(hit)[0] if hit else None


def _find_parent_in_found(concept_id: str, found_set: Set[str]) -> Tuple[Optional[str], int]:
    """
    Vérifie si un parent (ancêtre) du concept attendu est dans found_set.
    Retourne (parent_id, distance) du parent le plus proche trouvé, ou (None, 0).
    Distance 1 = parent direct (→ 2/3), distance 2+ = grand-parent (→ 1/3).
    """
    parents = _get_all_parents_recursive(concept_id, max_depth=3)
    best_parent = None
    best_dist = 999
    for pid, dist in parents.items():
        if pid in found_set and dist < best_dist:
            best_parent = pid
            best_dist = dist
    return (best_parent, best_dist) if best_parent else (None, 0)


# ---------------------------------------------------------------------------
# Recursive sub-require scoring
# ---------------------------------------------------------------------------

def _score_sub_require(
    concept_id: str,
    found_set: Set[str],
    depth: int = 0,
    max_depth: int = 2,
) -> float:
    """
    Score récursif d'un concept require qui n'est pas directement trouvé.

    Vérifie si ses propres requires / qualifiers / supports sont partiellement
    satisfaits. Retourne un score entre 0 et 1.

    Ex : QRS_NORMAL requires [QRS_FINS, ABSENCE_D_ONDE_Q_PATHOLOGIQUE]
         Si QRS_FINS trouvé → score = 0.5 (1/2 requires)
    """
    if depth >= max_depth:
        return 0.0

    onto = _get_ontology_v2()
    concepts = onto["concepts"]
    c = concepts.get(concept_id, {})

    # Check requires
    requires = c.get("requires", [])
    if requires:
        credit = 0.0
        for r in requires:
            nr = normalize_key(r)
            if nr in found_set:
                credit += 1.0
            else:
                credit += _score_sub_require(nr, found_set, depth + 1, max_depth)
        return credit / len(requires)

    # Check qualifiers
    qualifiers = list(c.get("has_qualifiers", []))
    for qfam in c.get("has_qualifier_families", []):
        nqf = normalize_key(qfam)
        if nqf not in qualifiers:
            qualifiers.append(nqf)
        for child in _get_all_children_recursive(nqf, max_depth=3):
            if child not in qualifiers:
                qualifiers.append(child)
    qual_found = [q for q in qualifiers if normalize_key(q) in found_set]
    if qual_found:
        return 2.0 / 3.0

    # Check supports
    supports = c.get("supports", [])
    sup_found = [s for s in supports if normalize_key(s) in found_set]
    if sup_found:
        return 1.0 / 3.0

    return 0.0


# ---------------------------------------------------------------------------
# Score d'un concept golden
# ---------------------------------------------------------------------------

def _score_one_concept(
    expected_id: str,
    found_set: Set[str],
) -> ConceptScore:
    """
    Calcule le score d'UN concept golden par rapport aux found_ids.
    """
    onto = _get_ontology_v2()
    concepts = onto["concepts"]

    neid = normalize_key(expected_id)
    c = concepts.get(neid, {})
    cname = c.get("concept_name", neid)

    cs = ConceptScore(concept_id=neid, concept_name=cname)

    # ── 0. Vérifier les exclusions d'abord ─────────────────────────
    excluded_by = _check_excludes(neid, found_set)
    if excluded_by:
        cs.match_type = "excluded"
        cs.score = 0.0
        cs.excluded_by = excluded_by
        cs.detail = f"Exclu par {excluded_by}"
        return cs

    # ── 1. Trouvé exact ? ──────────────────────────────────────────
    if neid in found_set:
        cs.match_type = "exact"
        cs.score = 1.0
        cs.detail = "Trouvé exact"
        return cs

    # ── 1b. Un enfant (plus spécifique) trouvé ? → score complet ──
    child_hit = _find_child_in_found(neid, found_set)
    if child_hit:
        cs.match_type = "exact"
        cs.score = 1.0
        cs.detail = f"Enfant trouvé: {child_hit}"
        return cs

    # ── 1c. Un parent (plus générique) trouvé ? ───────────────────
    #   parent +1 (direct) → 2/3,  parent +2 (éloigné) → 1/3
    parent_hit, parent_dist = _find_parent_in_found(neid, found_set)
    if parent_hit:
        if parent_dist <= 1:
            parent_score = 2.0 / 3.0
            cs.match_type = "qualifier"
            cs.detail = f"Parent direct trouvé: {parent_hit} (dist={parent_dist})"
        else:
            parent_score = 1.0 / 3.0
            cs.match_type = "support"
            cs.detail = f"Parent éloigné trouvé: {parent_hit} (dist={parent_dist})"
        # On garde ce score comme plancher, mais on continue à vérifier
        # requires/qualifiers/supports qui pourraient donner mieux
        # (on ne cumule pas, on prend le max)
        parent_cs_score = round(parent_score, 4)
    else:
        parent_cs_score = 0.0

    # ── 2. A des requires_findings ? ───────────────────────────────
    requires = c.get("requires", [])
    if requires:
        satisfied = []
        missing = []
        partial_credit = 0.0

        for r in requires:
            nr = normalize_key(r)
            if nr in found_set:
                satisfied.append(r)
                partial_credit += 1.0
            else:
                # Recursive : vérifier si ce require est lui-même partiellement satisfait
                sub_score = _score_sub_require(nr, found_set, depth=1)
                if sub_score > 0:
                    satisfied.append(f"{r}({sub_score:.0%})")
                    partial_credit += sub_score
                else:
                    missing.append(r)

        ratio = partial_credit / len(requires)
        req_score = round(ratio, 4)

        # Prendre le max entre requires et parent hierarchy
        if req_score >= parent_cs_score:
            cs.match_type = "requires"
            cs.score = req_score
            cs.requires_total = len(requires)
            cs.requires_found = len(satisfied)
            cs.requires_satisfied = satisfied
            cs.requires_missing = missing
            cs.detail = f"{partial_credit:.1f}/{len(requires)} requires"
        elif parent_cs_score > 0:
            cs.score = parent_cs_score
            # match_type and detail already set from parent block above
            if parent_cs_score >= 2.0 / 3.0:
                cs.match_type = "qualifier"
            else:
                cs.match_type = "support"
        else:
            cs.match_type = "requires"
            cs.score = req_score
            cs.requires_total = len(requires)
            cs.requires_found = len(satisfied)
            cs.requires_satisfied = satisfied
            cs.requires_missing = missing
            cs.detail = f"{partial_credit:.1f}/{len(requires)} requires"
        return cs

    # ── 3. has_qualifiers trouvés ? ────────────────────────────────
    qualifiers = list(c.get("has_qualifiers", []))
    # Étendre avec has_qualifier_families (le concept lui-même + ses enfants)
    for qfam in c.get("has_qualifier_families", []):
        nqf = normalize_key(qfam)
        if nqf not in qualifiers:
            qualifiers.append(nqf)
        for child in _get_all_children_recursive(nqf, max_depth=3):
            if child not in qualifiers:
                qualifiers.append(child)
    qual_found = [q for q in qualifiers if normalize_key(q) in found_set]
    if qual_found:
        qual_score = 2.0 / 3.0
        if qual_score >= parent_cs_score:
            cs.match_type = "qualifier"
            cs.score = qual_score
            cs.qualifiers_found = qual_found
            cs.detail = f"Qualifiers: {', '.join(qual_found)}"
        else:
            cs.score = parent_cs_score
        return cs

    # ── 4. supports trouvés ? ──────────────────────────────────────
    supports = c.get("supports", [])
    sup_found = [s for s in supports if normalize_key(s) in found_set]
    if sup_found:
        sup_score = 1.0 / 3.0
        if sup_score >= parent_cs_score:
            cs.match_type = "support"
            cs.score = sup_score
            cs.supports_found = sup_found
            cs.detail = f"Supports: {', '.join(sup_found)}"
        else:
            cs.score = parent_cs_score
        return cs

    # ── 5. Parent hierarchy comme filet de sécurité ────────────────
    if parent_cs_score > 0:
        cs.score = parent_cs_score
        # match_type and detail already set from parent block
        return cs

    # ── 6. Rien ────────────────────────────────────────────────────
    cs.match_type = "missed"
    cs.score = 0.0
    cs.detail = "Non trouvé"
    return cs


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def score_student_response_v3(
    found_ids: List[str],
    expected_ids: List[str],
    absent_ids: Optional[List[str]] = None,
) -> ScoringResultV3:
    """
    Scoring V3 ontologique.

    Args:
        found_ids:    IDs des concepts trouvés par le pipeline NER (présents uniquement)
        expected_ids: IDs des concepts attendus (golden set)
        absent_ids:   IDs des concepts avec statut "absent" dans le NER.
                      Convertis en concepts positifs via le mapping ontologique.

    Returns:
        ScoringResultV3 avec le détail par concept et le score global.
    """
    result = ScoringResultV3()

    if not expected_ids:
        return result

    # Normaliser les found_ids
    found_set = {normalize_key(fid) for fid in found_ids}

    # Convertir les absents en positifs et les ajouter aux found_ids
    if absent_ids:
        conversions = convert_absents_to_positive(absent_ids)
        for absent_id, positive_id in conversions:
            found_set.add(positive_id)
            result.negation_conversions.append((absent_id, positive_id))
            logger.debug("Negation: absent(%s) → +%s", absent_id, positive_id)

    # Scorer chaque concept golden
    for eid in expected_ids:
        cs = _score_one_concept(eid, found_set)
        result.concept_scores.append(cs)

    # Agrégation : chaque concept vaut 1/N
    n = len(expected_ids)
    result.max_possible_score = float(n)
    result.total_score = sum(cs.score for cs in result.concept_scores)
    result.score_pct = round((result.total_score / n) * 100, 1) if n > 0 else 0.0

    # Compteurs
    for cs in result.concept_scores:
        if cs.match_type == "exact":
            result.n_exact += 1
        elif cs.match_type == "requires":
            result.n_requires += 1
        elif cs.match_type == "qualifier":
            result.n_qualifier += 1
        elif cs.match_type == "support":
            result.n_support += 1
        elif cs.match_type == "excluded":
            result.n_excluded += 1
        else:
            result.n_missed += 1

    return result


# ---------------------------------------------------------------------------
# Format lisible (debug)
# ---------------------------------------------------------------------------

def format_v3_summary(result: ScoringResultV3) -> str:
    """Résumé lisible du scoring V3."""
    lines = [
        f"Score V3 : {result.score_pct:.1f}% ({result.total_score:.2f}/{result.max_possible_score:.0f})",
        f"  exact={result.n_exact} requires={result.n_requires} "
        f"qualifier={result.n_qualifier} support={result.n_support} "
        f"excluded={result.n_excluded} missed={result.n_missed}",
    ]
    if result.negation_conversions:
        lines.append(f"  négations converties: {len(result.negation_conversions)}")
        for absent_id, positive_id in result.negation_conversions:
            lines.append(f"    absent({absent_id}) → +{positive_id}")
    lines.append("")
    for cs in result.concept_scores:
        icon = {
            "exact": "✅", "requires": "📊", "qualifier": "🔶",
            "support": "🔹", "excluded": "🚫", "missed": "❌",
        }.get(cs.match_type, "?")
        lines.append(
            f"  {icon} {cs.concept_name:40s} → {cs.score:.2f}  [{cs.match_type}] {cs.detail}"
        )
    return "\n".join(lines)
