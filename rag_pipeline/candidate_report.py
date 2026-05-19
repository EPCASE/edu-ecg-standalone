"""
📋 Module de Rapport Candidat — Feedback structuré après évaluation (V3)
=========================================================================
Orchestre le pipeline complet (Briques 2→5) sur le texte d'un candidat
et génère un rapport pédagogique clair :

  1. 🔍 Analyse du texte — concepts extraits par le pipeline IA
  2. 📊 Note & explication — scoring V3 ontologique (requires/qualifier/support/excludes)
  3. 📝 Éléments descriptifs — concepts vrais ajoutés par le candidat (découvertes)

Utilisation :
    from candidate_report import generate_candidate_report, format_report_text

    report = generate_candidate_report(
        texte_etudiant="fibrillation atriale qrs fins",
        golden_ids=["FIBRILLATION_ATRIALE", "REPOLARISATION_PRÉCOCE"],
        diagnostic_principal="Fibrillation atriale",
    )
    print(format_report_text(report))

Auteur : BMad Team
Date   : 2026-04-06  (V3 — scoring ontologique)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ner_extractor import extract_clinical_terms, ClinicalEntity
from hybrid_search import HybridSearchEngine
from neurosymbolic_judge import resolve_term_to_ontology
from scoring_v3 import (
    score_student_response_v3,
    ScoringResultV3,
    ConceptScore,
    build_negation_map,
)
from semantic_layer import get_concept, normalize_key, _get_ontology_v2
from pedagogical_feedback import (
    generate_pedagogical_feedback,
    format_feedback_html,
    PedagogicalFeedback,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Filet de sécurité post-NER : détection et correction de la négation
# ──────────────────────────────────────────────────────────────────────────────

# Patterns de négation français courants dans les comptes-rendus ECG
_NEG_PREFIX_RE = re.compile(
    r"^(?:pas\s+(?:de\s+|d[''])|sans\s+|absence\s+(?:de\s+|d[''])|aucun(?:e)?\s+"
    r"|ni\s+|élimine\s+|n[''](?:est|a)\s+pas\s+)",
    re.IGNORECASE,
)


def _fix_negation(entite: ClinicalEntity) -> ClinicalEntity:
    """
    Filet de sécurité : si le NER a raté la négation, on la détecte
    via regex et on corrige statut + terme_brut.

    Deux cas couverts :
      1. terme_brut = "Pas de trouble de repolarisation" + statut="present"
         → terme_brut = "trouble de repolarisation", statut = "absent"
      2. terme_brut = "trouble de repolarisation" + statut="present"
         mais contexte_phrase contient "Pas de trouble de repolarisation"
         → statut = "absent"
    """
    # Cas 1 : la négation est dans le terme_brut lui-même
    m = _NEG_PREFIX_RE.match(entite.terme_brut)
    if m:
        cleaned = entite.terme_brut[m.end():].strip()
        if cleaned:
            logger.info(
                f"🔧 Fix négation (terme_brut) : "
                f"'{entite.terme_brut}' [{entite.statut}] → '{cleaned}' [absent]"
            )
            entite.terme_brut = cleaned
            entite.statut = "absent"
            return entite

    # Cas 2 : le terme_brut est propre mais le contexte contient la négation
    if entite.statut == "present":
        # Chercher "pas de <terme>" / "sans <terme>" dans le contexte
        terme_esc = re.escape(entite.terme_brut)
        neg_in_ctx = re.search(
            r"(?:pas\s+(?:de\s+|d[''])|sans\s+|absence\s+(?:de\s+|d[''])|aucun(?:e)?\s+)"
            + terme_esc,
            entite.contexte_phrase,
            re.IGNORECASE,
        )
        if neg_in_ctx:
            logger.info(
                f"🔧 Fix négation (contexte) : "
                f"'{entite.terme_brut}' [{entite.statut}] → [absent]  "
                f"(trouvé dans contexte : '{neg_in_ctx.group()}')"
            )
            entite.statut = "absent"

    return entite

@dataclass
class ExtractedConcept:
    """Un concept extrait du texte du candidat par le pipeline IA."""
    terme_brut: str
    statut: str                # present / absent / hypothese
    ontology_id: str           # ID résolu ou "NONE"
    concept_name: str          # Nom canonique dans l'ontologie
    method: str                # coupe_circuit / juge_llm / fallback_subterm / no_candidates
    justification: str         # Explication de la résolution
    # --- Métriques de confiance (NEW) ---
    top_k_candidats: list = field(default_factory=list)   # Top-K candidats avec scores
    llm_confiance: int = -1    # Confiance LLM auto-évaluée (0-100), -1 si coupe-circuit


@dataclass
class ValidantDetail:
    """Détail du scoring V3 pour un concept golden attendu."""
    golden_name: str
    golden_id: str
    found: bool
    score_pct: float           # 0..100
    match_type: str            # exact / requires / qualifier / support / excluded / missed
    detail: str                # Détail V3 (ex: "2/3 requires")
    explication: str           # Phrase d'explication pour le candidat
    # V3 specifics
    requires_satisfied: list = field(default_factory=list)
    requires_missing: list = field(default_factory=list)
    qualifiers_found: list = field(default_factory=list)
    supports_found: list = field(default_factory=list)
    excluded_by: str = ""


@dataclass
class DescripteurDetail:
    """Détail pour un élément descripteur attendu."""
    golden_name: str
    golden_id: str
    found: bool
    match_type: str


@dataclass
class DecouverteDetail:
    """Un concept trouvé par le candidat, vrai, mais non exigé par le barème."""
    concept_name: str
    ontology_id: str
    categorie: str
    statut: str


@dataclass
class CandidateReport:
    """Rapport complet d'évaluation pour un candidat."""
    # Méta
    diagnostic_principal: str
    texte_etudiant: str
    latence_s: float
    erreur: Optional[str] = None
    commentaire_correcteur: str = ""

    # Section 1 : Concepts extraits
    concepts_extraits: List[ExtractedConcept] = field(default_factory=list)

    # Section 2 : Note & explication
    score_final_pct: float = 0.0
    validant_details: List[ValidantDetail] = field(default_factory=list)
    nb_validants_trouves: int = 0
    nb_validants_attendus: int = 0

    # Section 3 : Descripteurs
    descripteur_details: List[DescripteurDetail] = field(default_factory=list)
    nb_descripteurs_trouves: int = 0
    nb_descripteurs_attendus: int = 0

    # Section 4 : Découvertes additionnelles
    decouvertes: List[DecouverteDetail] = field(default_factory=list)

    # Section 5 : Feedback pédagogique (cours SFC)
    feedback_pedagogique: Optional[PedagogicalFeedback] = None

    # Statistiques méthodes
    n_coupe_circuit: int = 0
    n_juge_llm: int = 0
    n_fallback: int = 0
    n_no_candidates: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Moteur de recherche (singleton module-level)
# ──────────────────────────────────────────────────────────────────────────────

_engine: Optional[HybridSearchEngine] = None


def _get_engine() -> HybridSearchEngine:
    global _engine
    if _engine is None:
        _engine = HybridSearchEngine()
    return _engine


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _explain_match_type_v3(cs: ConceptScore) -> str:
    """Génère une explication lisible V3 pour le candidat."""
    name = cs.concept_name
    pct = f"{cs.score * 100:.0f}%"

    if cs.match_type == "exact":
        return f"✅ Vous avez identifié « {name} » — correspondance exacte ({pct})."

    if cs.match_type == "requires":
        sat = cs.requires_satisfied or []
        mis = cs.requires_missing or []
        found_str = ", ".join(sat) if sat else "aucun"
        miss_str = ", ".join(mis) if mis else "aucun"
        return (
            f"� « {name} » partiellement validé ({pct}) — "
            f"critères trouvés : {found_str} ; manquants : {miss_str}."
        )

    if cs.match_type == "qualifier":
        quals = ", ".join(cs.qualifiers_found) if cs.qualifiers_found else "?"
        return (
            f"🔶 « {name} » reconnu via qualifier ({pct}) — "
            f"élément(s) qualifiant(s) trouvé(s) : {quals}."
        )

    if cs.match_type == "support":
        sups = ", ".join(cs.supports_found) if cs.supports_found else "?"
        return (
            f"� « {name} » partiellement supporté ({pct}) — "
            f"élément(s) de soutien : {sups}."
        )

    if cs.match_type == "excluded":
        return (
            f"� « {name} » exclu ({pct}) — "
            f"la présence de « {cs.excluded_by} » contredit ce diagnostic."
        )

    # missed
    return f"❌ « {name} » n'a pas été identifié dans votre réponse."


def _match_type_label_v3(mt: str, score: float) -> str:
    """Label court lisible pour un match_type V3."""
    pct = f"{score * 100:.0f}%"
    labels = {
        "exact": f"Exact ({pct})",
        "requires": f"Requires ({pct})",
        "qualifier": f"Qualifier ({pct})",
        "support": f"Support ({pct})",
        "excluded": f"Exclu ({pct})",
        "missed": f"Manquant ({pct})",
    }
    return labels.get(mt, f"{mt} ({pct})")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def generate_candidate_report(
    texte_etudiant: str,
    golden_names: List[str] = None,
    golden_ids: List[str] = None,
    golden_roles: List[str] = None,
    diagnostic_principal: str = "",
    moteur: Optional[HybridSearchEngine] = None,
    with_feedback: bool = True,
    commentaire_correcteur: str = "",
) -> CandidateReport:
    """
    Exécute le pipeline complet et construit un CandidateReport (V3).

    Args:
        texte_etudiant:       Texte libre du candidat.
        golden_names:         Noms des concepts attendus (golden set). Optionnel si golden_ids fournis.
        golden_ids:           IDs ontologiques attendus (golden set).
        golden_roles:         "validant" ou "descripteur" pour chaque concept.
                              Si None, tous sont considérés comme validants.
        diagnostic_principal: Diagnostic principal du cas (pour affichage).
        moteur:               HybridSearchEngine pré-initialisé (optionnel).
        with_feedback:        Si True (défaut), génère le feedback pédagogique GPT.
                              Mettre à False pour les benchmarks/tests rapides.
        commentaire_correcteur: Commentaire libre du correcteur humain.

    Returns:
        CandidateReport complet.
    """
    golden_ids = golden_ids or []
    golden_names = golden_names or []
    golden_roles = golden_roles or ["validant"] * len(golden_ids)

    # Résoudre les noms depuis l'ontologie V2 si non fournis
    if golden_ids and not golden_names:
        for gid in golden_ids:
            c = get_concept(normalize_key(gid))
            golden_names.append(c.get("concept_name", gid) if c else gid)

    engine = moteur or _get_engine()

    report = CandidateReport(
        diagnostic_principal=diagnostic_principal,
        texte_etudiant=texte_etudiant,
        latence_s=0.0,
        commentaire_correcteur=commentaire_correcteur,
    )

    if not texte_etudiant or texte_etudiant.strip() in ("", "nan"):
        report.erreur = "Texte vide"
        return report

    t0 = time.time()

    try:
        # ═══════════════════════════════════════════════════════════════
        # Brique 2 : Extraction NER
        # ═══════════════════════════════════════════════════════════════
        extraction = extract_clinical_terms(texte_etudiant)

        # ═══════════════════════════════════════════════════════════════
        # Briques 3 + 4 : Recherche hybride + Juge neurosymbolique
        # ═══════════════════════════════════════════════════════════════
        student_matched_ids: Dict[str, str] = {}  # id → statut
        methods: List[str] = []

        for entite in extraction.entites:
            # Filet de sécurité : corriger la négation si le NER l'a ratée
            entite = _fix_negation(entite)

            candidats = engine.search_top_k(entite.terme_brut)
            resolution = resolve_term_to_ontology(
                entite.terme_brut, entite.contexte_phrase, candidats
            )
            matched_id = resolution["ontology_id"]
            method = resolution["method"]
            methods.append(method)

            concept = ExtractedConcept(
                terme_brut=entite.terme_brut,
                statut=entite.statut,
                ontology_id=matched_id,
                concept_name=resolution.get("concept_name", ""),
                method=method,
                justification=resolution.get("justification", ""),
                top_k_candidats=resolution.get("top_k_candidats", []),
                llm_confiance=resolution.get("llm_confiance", -1),
            )
            report.concepts_extraits.append(concept)

            if matched_id != "NONE":
                student_matched_ids[matched_id] = entite.statut

        # Stats méthodes
        report.n_coupe_circuit = methods.count("coupe_circuit")
        report.n_juge_llm = methods.count("juge_llm")
        report.n_fallback = methods.count("fallback_subterm")
        report.n_no_candidates = methods.count("no_candidates")

        # ═══════════════════════════════════════════════════════════════
        # Brique 5 : Scoring V3 ontologique
        # ═══════════════════════════════════════════════════════════════
        # Séparer présents et absents
        found_ids = [oid for oid, st in student_matched_ids.items()
                     if st in ("present", "hypothese")]
        absent_ids = [oid for oid, st in student_matched_ids.items()
                      if st == "absent"]

        # Seuls les validants sont scorés en V3
        validant_ids = [gid for gid, role in zip(golden_ids, golden_roles)
                        if role == "validant"]

        v3_result: ScoringResultV3 = score_student_response_v3(
            found_ids=found_ids,
            expected_ids=validant_ids,
            absent_ids=absent_ids,
        )

        report.score_final_pct = v3_result.score_pct

        # ─── Construire le détail des validants (V3) ─────────────────
        report.nb_validants_attendus = len(validant_ids)
        report.nb_validants_trouves = v3_result.n_exact + v3_result.n_requires + v3_result.n_qualifier + v3_result.n_support

        # Index golden_id → golden_name
        id_to_golden_name = {}
        for gname, gid, role in zip(golden_names, golden_ids, golden_roles):
            if role == "validant":
                id_to_golden_name[normalize_key(gid)] = gname

        for cs in v3_result.concept_scores:
            gname = id_to_golden_name.get(cs.concept_id, cs.concept_name)
            found = cs.match_type not in ("missed", "excluded")
            report.validant_details.append(ValidantDetail(
                golden_name=gname,
                golden_id=cs.concept_id,
                found=found,
                score_pct=round(cs.score * 100, 1),
                match_type=cs.match_type,
                detail=cs.detail,
                explication=_explain_match_type_v3(cs),
                requires_satisfied=cs.requires_satisfied,
                requires_missing=cs.requires_missing,
                qualifiers_found=cs.qualifiers_found,
                supports_found=cs.supports_found,
                excluded_by=cs.excluded_by,
            ))

        # ─── Construire le détail des descripteurs ───────────────────
        descripteur_ids = [gid for gid, role in zip(golden_ids, golden_roles)
                          if role == "descripteur"]
        found_set = {normalize_key(fid) for fid in found_ids}
        # Ajouter les positifs issus de négations converties
        for _, pos_id in v3_result.negation_conversions:
            found_set.add(pos_id)

        report.nb_descripteurs_attendus = len(descripteur_ids)
        n_desc_found = 0
        for gname, gid, role in zip(golden_names, golden_ids, golden_roles):
            if role != "descripteur":
                continue
            nid = normalize_key(gid)
            # Exact match
            found = nid in found_set
            match_type = "exact" if found else "missed"
            # Child match (enfant plus spécifique trouvé → compte comme trouvé)
            if not found:
                from scoring_v3 import _find_child_in_found, _find_parent_in_found
                child_hit = _find_child_in_found(nid, found_set)
                if child_hit:
                    found = True
                    match_type = "exact"  # enfant = credit complet
                else:
                    # Parent match (parent trouvé → partiel)
                    parent_hit, parent_dist = _find_parent_in_found(nid, found_set)
                    if parent_hit:
                        found = True
                        match_type = "qualifier" if parent_dist <= 1 else "support"
            if found:
                n_desc_found += 1
            report.descripteur_details.append(DescripteurDetail(
                golden_name=gname,
                golden_id=gid,
                found=found,
                match_type=match_type,
            ))
        report.nb_descripteurs_trouves = n_desc_found

        # ─── Découvertes additionnelles ──────────────────────────────
        golden_id_set = {normalize_key(gid) for gid in golden_ids}
        onto = _get_ontology_v2()
        for fid in found_set - golden_id_set:
            c = onto["concepts"].get(fid, {})
            if c:
                report.decouvertes.append(DecouverteDetail(
                    concept_name=c.get("concept_name", fid),
                    ontology_id=fid,
                    categorie=c.get("type", "DESCRIPTEUR_ECG"),
                    statut="present",
                ))

        # ═══════════════════════════════════════════════════════════════
        # Brique 6 : Feedback pédagogique (cours SFC, Item 231)
        # ═══════════════════════════════════════════════════════════════
        if with_feedback:
            try:
                report.feedback_pedagogique = generate_pedagogical_feedback(report)
            except Exception as fb_err:
                logger.warning(f"Feedback pédagogique indisponible : {fb_err}")

    except Exception as e:
        report.erreur = str(e)[:200]

    report.latence_s = round(time.time() - t0, 2)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Formatage texte (terminal / console)
# ──────────────────────────────────────────────────────────────────────────────

def format_report_text(report: CandidateReport) -> str:
    """
    Formate un CandidateReport en texte lisible (terminal / console).
    """
    lines: List[str] = []
    W = 80

    lines.append("═" * W)
    lines.append(f"📋 RAPPORT D'ÉVALUATION — {report.diagnostic_principal}")
    lines.append("═" * W)

    if report.erreur:
        lines.append(f"\n⚠️  Erreur : {report.erreur}")
        return "\n".join(lines)

    # ─── Section 1 : Concepts extraits ────────────────────────────────────
    lines.append(f"\n{'─'*W}")
    lines.append(f"🔍 SECTION 1 — Analyse de votre texte ({len(report.concepts_extraits)} concepts identifiés)")
    lines.append(f"{'─'*W}")
    lines.append(f'   Votre texte : « {report.texte_etudiant} »\n')

    for i, c in enumerate(report.concepts_extraits, 1):
        statut_icon = {"present": "✓", "absent": "✗", "hypothese": "?"}.get(c.statut, "·")
        if c.ontology_id != "NONE":
            lines.append(
                f"   {i}. [{statut_icon}] « {c.terme_brut} »  →  {c.concept_name}"
                f"  ({c.method})"
            )
        else:
            lines.append(
                f"   {i}. [{statut_icon}] « {c.terme_brut} »  →  ⚠️ non résolu"
            )

    # ─── Section 2 : Note & explication ───────────────────────────────────
    lines.append(f"\n{'─'*W}")
    lines.append(
        f"📊 SECTION 2 — Votre note : {report.score_final_pct:.1f}% "
        f"({report.nb_validants_trouves}/{report.nb_validants_attendus} diagnostics validants)"
    )
    lines.append(f"{'─'*W}")

    # Rappel du barème V3
    lines.append(f"   Barème V3 : Exact=100% | Requires=proportionnel | Qualifier=67% | Support=33% | Exclu/Manqué=0%\n")

    for vd in report.validant_details:
        score_str = f"{vd.score_pct:5.1f}%"
        lines.append(f"   {vd.explication}")
        if vd.match_type not in ("exact", "missed", "excluded"):
            lines.append(f"          → Score pour ce concept : {score_str}")
        elif vd.match_type in ("missed", "excluded"):
            lines.append(f"          → Score pour ce concept : {score_str}")

    # Score final
    lines.append(f"\n   ══ NOTE FINALE : {report.score_final_pct:.1f}% ══")

    # ─── Section 3 : Descripteurs ─────────────────────────────────────────
    if report.nb_descripteurs_attendus > 0:
        lines.append(f"\n{'─'*W}")
        lines.append(
            f"📝 SECTION 3 — Éléments descriptifs "
            f"({report.nb_descripteurs_trouves}/{report.nb_descripteurs_attendus} identifiés)"
        )
        lines.append(f"{'─'*W}")
        lines.append(
            f"   Ces éléments font partie du diagnostic mais ne sont pas notés :\n"
        )

        for dd in report.descripteur_details:
            if dd.found:
                lines.append(f"   ✅ « {dd.golden_name} » — identifié")
            else:
                lines.append(f"   ⬜ « {dd.golden_name} » — non mentionné")

    # ─── Section 4 : Découvertes ──────────────────────────────────────────
    if report.decouvertes:
        lines.append(f"\n{'─'*W}")
        lines.append(
            f"🟢 SECTION 4 — Découvertes additionnelles "
            f"({len(report.decouvertes)} concepts vrais, non exigés)"
        )
        lines.append(f"{'─'*W}")
        lines.append(
            f"   Vous avez identifié des éléments cliniquement pertinents\n"
            f"   au-delà du barème strict. Ils ne rapportent pas de points\n"
            f"   mais montrent la qualité de votre lecture :\n"
        )

        for dec in report.decouvertes:
            cat_label = dec.categorie.replace("_", " ").capitalize()
            lines.append(f"   🟢 {dec.concept_name}  ({cat_label})")

    # ─── Section 5 : Feedback pédagogique ─────────────────────────────
    if report.feedback_pedagogique and report.feedback_pedagogique.texte:
        lines.append(f"\n{'─'*W}")
        lines.append(
            f"🎓 SECTION 5 — Commentaire pédagogique (Item 231, SFC)"
        )
        lines.append(f"{'─'*W}")
        if report.feedback_pedagogique.has_critical_miss:
            lines.append("   ⚠️  ATTENTION : un concept de Rang A (indispensable) a été manqué.\n")
        lines.append(report.feedback_pedagogique.texte)

    # ─── Footer ───────────────────────────────────────────────────────────
    lines.append(f"\n{'═'*W}")
    lines.append(f"⏱️  Temps d'analyse : {report.latence_s:.1f}s")
    lines.append("═" * W)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Formatage HTML (pour Streamlit / notebook)
# ──────────────────────────────────────────────────────────────────────────────

def format_report_html(report: CandidateReport) -> str:
    """
    Formate un CandidateReport en HTML stylisé (dark theme).
    Compatible Streamlit (st.markdown) et IPython (display(HTML(...))).
    """
    if report.erreur:
        return f'<div style="color:#ff6b6b; padding:20px;">⚠️ Erreur : {report.erreur}</div>'

    # Couleur du score
    score = report.score_final_pct
    if score >= 90:
        score_color = "#4CAF50"
        score_emoji = "🎉"
        score_label = "Excellent"
    elif score >= 70:
        score_color = "#FF9800"
        score_emoji = "👍"
        score_label = "Bien"
    elif score >= 50:
        score_color = "#FF5722"
        score_emoji = "📚"
        score_label = "À améliorer"
    else:
        score_color = "#F44336"
        score_emoji = "💪"
        score_label = "À retravailler"

    html_parts: List[str] = []

    # ─── Container ────────────────────────────────────────────────────
    html_parts.append(f"""
    <div style="background:#1e1e1e; color:#e0e0e0; padding:24px; border-radius:12px;
                font-family:'Segoe UI', system-ui, sans-serif; line-height:1.6;">
    """)

    # ─── Header : Score ───────────────────────────────────────────────
    html_parts.append(f"""
    <div style="text-align:center; margin-bottom:24px;">
        <div style="font-size:14px; color:#999; text-transform:uppercase; letter-spacing:2px;">
            {report.diagnostic_principal}
        </div>
        <div style="font-size:64px; font-weight:bold; color:{score_color}; margin:8px 0;">
            {score:.0f}%
        </div>
        <div style="font-size:18px; color:{score_color};">
            {score_emoji} {score_label} — {report.nb_validants_trouves}/{report.nb_validants_attendus} diagnostics validants
        </div>
        <div style="font-size:12px; color:#666; margin-top:4px;">
            Barème V3 : Exact=100% | Requires=proportionnel | Qualifier=67% | Support=33%
        </div>
    </div>
    """)

    # ─── Section 1 : Concepts extraits ────────────────────────────────
    html_parts.append(f"""
    <div style="background:#252525; border-radius:8px; padding:16px; margin-bottom:16px;">
        <h3 style="color:#90CAF9; margin:0 0 12px 0; font-size:16px;">
            🔍 Analyse de votre texte — {len(report.concepts_extraits)} concepts identifiés
        </h3>
        <div style="background:#2d2d2d; padding:10px; border-radius:6px; margin-bottom:12px;
                    font-style:italic; color:#bbb;">
            « {report.texte_etudiant} »
        </div>
    """)

    for c in report.concepts_extraits:
        if c.ontology_id != "NONE":
            method_badge = {
                "coupe_circuit": "⚡",
                "juge_llm": "🧠",
                "fallback_subterm": "🔄",
            }.get(c.method, "·")
            html_parts.append(f"""
            <div style="padding:4px 0; border-bottom:1px solid #333;">
                <span style="color:#4CAF50;">●</span>
                <strong>« {c.terme_brut} »</strong>
                → <span style="color:#81C784;">{c.concept_name}</span>
                <span style="color:#666; font-size:12px; margin-left:8px;">{method_badge} {c.method}</span>
            </div>
            """)
        else:
            html_parts.append(f"""
            <div style="padding:4px 0; border-bottom:1px solid #333;">
                <span style="color:#9E9E9E;">●</span>
                <strong>« {c.terme_brut} »</strong>
                → <span style="color:#9E9E9E;">non résolu</span>
            </div>
            """)

    html_parts.append("</div>")

    # ─── Section 2 : Détail du scoring ────────────────────────────────
    html_parts.append(f"""
    <div style="background:#252525; border-radius:8px; padding:16px; margin-bottom:16px;">
        <h3 style="color:#FFD54F; margin:0 0 12px 0; font-size:16px;">
            📊 Détail de votre note — {report.score_final_pct:.1f}%
        </h3>
    """)

    for vd in report.validant_details:
        mt_colors = {
            "exact": ("#4CAF50", "✅"),
            "requires": ("#1565C0", "📊"),
            "qualifier": ("#FF9800", "🔶"),
            "support": ("#00BCD4", "🔹"),
            "excluded": ("#F44336", "�"),
            "missed": ("#9E9E9E", "❌"),
        }
        color, icon = mt_colors.get(vd.match_type, ("#9E9E9E", "❌"))

        score_bar_width = min(100, max(0, vd.score_pct))
        html_parts.append(f"""
        <div style="background:#2d2d2d; border-radius:6px; padding:10px; margin-bottom:8px;
                    border-left:4px solid {color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span>{icon} <strong style="color:{color};">{vd.golden_name}</strong></span>
                <span style="color:{color}; font-weight:bold; font-size:18px;">{vd.score_pct:.0f}%</span>
            </div>
            <div style="background:#1a1a1a; border-radius:4px; height:6px; margin:6px 0;">
                <div style="background:{color}; width:{score_bar_width}%; height:100%; border-radius:4px;"></div>
            </div>
            <div style="color:#aaa; font-size:13px;">{vd.explication}</div>""")
        # V3 detail: requires/qualifiers/supports
        if vd.requires_satisfied or vd.requires_missing:
            sat_html = ", ".join(vd.requires_satisfied) if vd.requires_satisfied else ""
            mis_html = ", ".join(vd.requires_missing) if vd.requires_missing else ""
            detail_parts = []
            if sat_html:
                detail_parts.append(f'<span style="color:#4CAF50;">✓ {sat_html}</span>')
            if mis_html:
                detail_parts.append(f'<span style="color:#F44336;">✗ {mis_html}</span>')
            html_parts.append(f'<div style="font-size:12px; color:#888; margin-top:4px; padding-left:8px;">{" | ".join(detail_parts)}</div>')
        elif vd.qualifiers_found:
            q_html = ", ".join(vd.qualifiers_found)
            html_parts.append(f'<div style="font-size:12px; color:#FF9800; margin-top:4px; padding-left:8px;">via qualifier: {q_html}</div>')
        elif vd.supports_found:
            s_html = ", ".join(vd.supports_found)
            html_parts.append(f'<div style="font-size:12px; color:#00BCD4; margin-top:4px; padding-left:8px;">via support: {s_html}</div>')
        elif vd.excluded_by:
            html_parts.append(f'<div style="font-size:12px; color:#F44336; margin-top:4px; padding-left:8px;">exclu par: {vd.excluded_by}</div>')
        html_parts.append("</div>")

    html_parts.append("</div>")

    # ─── Section 3 : Descripteurs ─────────────────────────────────────
    if report.nb_descripteurs_attendus > 0:
        html_parts.append(f"""
        <div style="background:#252525; border-radius:8px; padding:16px; margin-bottom:16px;">
            <h3 style="color:#CE93D8; margin:0 0 12px 0; font-size:16px;">
                📝 Éléments descriptifs — {report.nb_descripteurs_trouves}/{report.nb_descripteurs_attendus} identifiés
            </h3>
            <div style="color:#999; font-size:13px; margin-bottom:8px;">
                Ces éléments font partie du diagnostic mais ne sont pas notés.
            </div>
        """)

        for dd in report.descripteur_details:
            if dd.found:
                html_parts.append(f"""
                <div style="padding:4px 8px;">
                    <span style="color:#4CAF50;">✅</span> {dd.golden_name}
                </div>
                """)
            else:
                html_parts.append(f"""
                <div style="padding:4px 8px;">
                    <span style="color:#666;">⬜</span>
                    <span style="color:#888;">{dd.golden_name}</span>
                </div>
                """)

        html_parts.append("</div>")

    # ─── Section 4 : Découvertes ──────────────────────────────────────
    if report.decouvertes:
        html_parts.append(f"""
        <div style="background:#252525; border-radius:8px; padding:16px; margin-bottom:16px;">
            <h3 style="color:#00BCD4; margin:0 0 12px 0; font-size:16px;">
                🟢 Découvertes additionnelles — {len(report.decouvertes)} concepts vrais
            </h3>
            <div style="color:#999; font-size:13px; margin-bottom:8px;">
                Vous avez identifié des éléments cliniquement pertinents au-delà du barème.
                Ils ne rapportent pas de points mais montrent la qualité de votre lecture.
            </div>
        """)

        for dec in report.decouvertes:
            cat_label = dec.categorie.replace("_", " ").capitalize()
            html_parts.append(f"""
            <div style="padding:4px 8px; border-bottom:1px solid #333;">
                <span style="color:#00BCD4;">🟢</span> <strong>{dec.concept_name}</strong>
                <span style="color:#666; font-size:12px; margin-left:8px;">({cat_label})</span>
            </div>
            """)

        html_parts.append("</div>")

    # ─── Section 5 : Feedback pédagogique ─────────────────────────────
    if report.feedback_pedagogique and report.feedback_pedagogique.texte:
        html_parts.append(format_feedback_html(report.feedback_pedagogique))

    # ─── Footer ───────────────────────────────────────────────────────
    html_parts.append(f"""
    <div style="text-align:center; color:#666; font-size:12px; margin-top:8px;">
        ⏱️ Temps d'analyse : {report.latence_s:.1f}s
    </div>
    </div>
    """)

    return "\n".join(html_parts)


# ──────────────────────────────────────────────────────────────────────────────
# CLI — Test rapide
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    # Charger .env
    load_dotenv(Path(__file__).parent.parent / "ECG lecture" / ".env")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.chdir(str(Path(__file__).parent))

    # Cas test : FA avec réponse partielle
    report = generate_candidate_report(
        texte_etudiant="fibrillation atriale qrs fins tachycardie",
        golden_names=["Fibrillation atriale", "Repolarisation précoce"],
        golden_ids=["FIBRILLATION_ATRIALE", "REPOLARISATION_PRÉCOCE"],
        golden_roles=["validant", "descripteur"],
        diagnostic_principal="Fibrillation atriale",
    )

    print(format_report_text(report))
