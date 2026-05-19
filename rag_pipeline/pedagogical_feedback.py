"""
🎓 Feedback Pédagogique — Commentaire personnalisé basé sur le cours SFC
==========================================================================
Génère un feedback pédagogique pour un candidat après évaluation ECG,
en s'appuyant sur les extraits du cours SFC (Item 231 EDN).

Le feedback est :
  - Personnalisé selon les erreurs/réussites du candidat
  - Ancré dans le cours officiel (citations SFC)
  - Structuré : félicitations → erreurs avec rappels de cours → pièges → conseils

Utilisation :
    from pedagogical_feedback import generate_pedagogical_feedback
    from candidate_report import generate_candidate_report

    report = generate_candidate_report(...)
    feedback = generate_pedagogical_feedback(report)
    print(feedback.texte)

Auteur : BMad Team
Date   : 2026-02-28
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from openai import OpenAI

from edn_knowledge_base import (
    EDNEntry,
    get_edn_entry,
    get_edn_entries_for_ids,
    POINTS_CLES_GENERAUX,
)

logger = logging.getLogger(__name__)

# Type hints pour éviter les imports circulaires
# On importe CandidateReport au runtime uniquement
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from candidate_report import CandidateReport


# ──────────────────────────────────────────────────────────────────────────────
# Structures de données
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PedagogicalFeedback:
    """Feedback pédagogique structuré pour le candidat."""
    texte: str                          # Texte complet du feedback (markdown)
    rang_edn_manques: List[str]         # Rangs EDN des concepts manqués ("A", "B")
    concepts_cours_cites: List[str]     # Noms des concepts pour lesquels le cours est cité
    has_critical_miss: bool             # True si un concept rang A a été manqué
    erreur: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Construction du contexte de cours pour le prompt
# ──────────────────────────────────────────────────────────────────────────────

def _build_course_context(report) -> str:
    """
    Construit le contexte de cours pertinent à injecter dans le prompt GPT.
    Sélectionne uniquement les entrées EDN liées aux concepts du cas.
    """
    # Collecter tous les ontology_ids pertinents pour ce cas
    relevant_ids: Set[str] = set()

    # IDs des validants (attendus)
    for vd in report.validant_details:
        relevant_ids.add(vd.golden_id)

    # IDs des descripteurs (attendus)
    for dd in report.descripteur_details:
        relevant_ids.add(dd.golden_id)

    # IDs des concepts trouvés par le candidat
    for c in report.concepts_extraits:
        if c.ontology_id != "NONE":
            relevant_ids.add(c.ontology_id)

    # IDs des découvertes
    for dec in report.decouvertes:
        relevant_ids.add(dec.ontology_id)

    # Récupérer les entrées EDN (dédupliquées par objet)
    seen_entries: Set[int] = set()
    entries: List[EDNEntry] = []

    for oid in relevant_ids:
        entry = get_edn_entry(oid)
        if entry and id(entry) not in seen_entries:
            seen_entries.add(id(entry))
            entries.append(entry)

    if not entries:
        return "Aucun extrait de cours pertinent trouvé pour ce cas."

    # Construire le texte de contexte
    parts = []
    parts.append("=== EXTRAITS DU COURS SFC — Item 231 (EDN) ===\n")

    for entry in entries:
        rang_label = {"A": "RANG A (indispensable)", "B": "RANG B (important)", "C": "RANG C (complémentaire)"}
        parts.append(f"--- {entry.titre_cours} [{rang_label.get(entry.rang_edn, entry.rang_edn)}] ---")
        parts.append(f"Extrait : {entry.extrait_cours}")
        if entry.points_cles:
            parts.append("Points clés :")
            for pc in entry.points_cles:
                parts.append(f"  • {pc}")
        if entry.pieges_classiques:
            parts.append("Pièges classiques :")
            for piege in entry.pieges_classiques:
                parts.append(f"  ⚠️ {piege}")
        parts.append("")

    return "\n".join(parts)


def _build_student_summary(report) -> str:
    """Construit un résumé structuré de la performance du candidat."""
    parts = []

    parts.append(f"DIAGNOSTIC PRINCIPAL DU CAS : {report.diagnostic_principal}")
    parts.append(f"TEXTE DU CANDIDAT : « {report.texte_etudiant} »")
    parts.append(f"SCORE FINAL : {report.score_final_pct:.1f}%")
    parts.append("")

    # Validants
    parts.append("DIAGNOSTICS VALIDANTS (notés) :")
    for vd in report.validant_details:
        status = "TROUVÉ" if vd.found else "MANQUÉ"
        entry = get_edn_entry(vd.golden_id)
        rang = f" [Rang EDN : {entry.rang_edn}]" if entry else ""
        if vd.match_type == "exact":
            parts.append(f"  ✅ {vd.golden_name} — {status} (exact, 100%){rang}")
        elif vd.match_type == "requires":
            sat = ", ".join(vd.requires_satisfied) if hasattr(vd, 'requires_satisfied') and vd.requires_satisfied else "?"
            parts.append(f"  � {vd.golden_name} — {status} (requires, {vd.score_pct:.0f}% — trouvés: {sat}){rang}")
        elif vd.match_type == "qualifier":
            quals = ", ".join(vd.qualifiers_found) if hasattr(vd, 'qualifiers_found') and vd.qualifiers_found else "?"
            parts.append(f"  🔶 {vd.golden_name} — {status} (qualifier, {vd.score_pct:.0f}% — via: {quals}){rang}")
        elif vd.match_type == "support":
            sups = ", ".join(vd.supports_found) if hasattr(vd, 'supports_found') and vd.supports_found else "?"
            parts.append(f"  🔹 {vd.golden_name} — {status} (support, {vd.score_pct:.0f}% — via: {sups}){rang}")
        elif vd.match_type == "excluded":
            excl = vd.excluded_by if hasattr(vd, 'excluded_by') and vd.excluded_by else "?"
            parts.append(f"  🚫 {vd.golden_name} — EXCLU (contredit par: {excl}){rang}")
        else:
            parts.append(f"  ❌ {vd.golden_name} — {status}{rang}")

    # Descripteurs
    if report.descripteur_details:
        parts.append("\nDESCRIPTEURS (non notés) :")
        for dd in report.descripteur_details:
            status = "identifié" if dd.found else "non mentionné"
            parts.append(f"  {'✅' if dd.found else '⬜'} {dd.golden_name} — {status}")

    # Découvertes
    if report.decouvertes:
        parts.append(f"\nDÉCOUVERTES ADDITIONNELLES ({len(report.decouvertes)} concepts vrais hors barème) :")
        for dec in report.decouvertes:
            parts.append(f"  🟢 {dec.concept_name} ({dec.categorie})")

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt système pour le feedback pédagogique
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un professeur de cardiologie bienveillant et pédagogue qui corrige l'interprétation ECG d'un étudiant en médecine préparant les EDN.

Tu disposes :
1. Du résultat de l'évaluation automatique (concepts trouvés/manqués, score V3 ontologique)
2. D'extraits du cours SFC officiel (Item 231 — Chapitre 15) pertinents pour ce cas
3. D'un éventuel commentaire du correcteur humain (expert) sur ce cas

Le scoring V3 utilise des relations ontologiques pour évaluer la réponse :
- **exact** (100%) : le concept attendu est directement identifié
- **requires** (proportionnel) : certains critères constitutifs du concept ont été trouvés
- **qualifier** (67%) : un élément qualifiant le concept a été identifié
- **support** (33%) : un élément de soutien indirect a été trouvé
- **excluded** (0%) : un concept contradictoire a été identifié, invalidant la réponse
- **missed** (0%) : rien de pertinent n'a été trouvé

Ta mission est de rédiger un COMMENTAIRE PÉDAGOGIQUE personnalisé en français, structuré OBLIGATOIREMENT en **2 parties** avec les titres exacts suivants :

---

## 1. Référence au cours

Pour les concepts clés de ce cas, cite la référence au cours SFC. Procède dans cet ordre de priorité :
1. **Concepts validants** (diagnostics notés) : trouvés ou manqués, cite le cours et le rang EDN
2. **Concepts descripteurs** (signes attendus) : s'ils sont manqués, rappelle le cours
3. **Concepts erronés** (excluded) : si un concept trouvé par l'étudiant est contradictoire avec un attendu, CITE-LE EXPLICITEMENT et explique le conflit en citant le cours

Pour chaque concept cité :
- Rappelle le **rang de connaissance EDN** (A = indispensable, B = important, C = complémentaire)
- Cite le cours SFC entre guillemets : 📖 « extrait du cours » — (Item 231, SFC)
- Si le concept est de rang A, insiste sur son caractère indispensable aux EDN

Sois CONCIS : ne cite que les concepts les plus importants (max 3-4 rappels de cours).

## 2. Votre interprétation

Analyse personnalisée COURTE de la copie de l'étudiant :
- Si le score est élevé (≥80%) : une phrase de félicitation courte et sincère
- Si le score n'est pas 100% : explique CLAIREMENT et BRIÈVEMENT pourquoi (quels concepts manqués/partiels), sans paraphraser la section 1
- Si des concepts sont **erronés (excluded)**, signale-le comme point d'attention
- Si des découvertes pertinentes ont été faites hors barème, mentionne-les brièvement (preuve de compétence)
- Si un **commentaire du correcteur** est fourni, intègre-le ici comme conseil expert : reprends-le, développe-le en 1-2 phrases en le reliant au cours SFC
- Si le commentaire du correcteur est vide et le score < 100%, donne un conseil concret en 1 phrase
- Si le score est 100%, encourage à aller plus loin (ex: "Vous pourriez approfondir...")

---

## Règles :
- Ton bienveillant mais exigeant, comme un bon PU qui veut que ses étudiants réussissent
- TOUJOURS citer le cours SFC quand tu fais un rappel (entre guillemets avec source)
- Respecter STRICTEMENT les 2 titres de section (## 1. Référence au cours / ## 2. Votre interprétation)
- Ne PAS répéter le score numérique (il est déjà affiché ailleurs)
- Rester CONCIS : 150-300 mots maximum au total
- Le commentaire est au format texte simple (pas de HTML), avec des emojis si pertinent
"""


# ──────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ──────────────────────────────────────────────────────────────────────────────

def generate_pedagogical_feedback(
    report,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    commentaire_correcteur: str = "",
) -> PedagogicalFeedback:
    """
    Génère un feedback pédagogique basé sur le cours SFC pour un CandidateReport.

    Args:
        report:      CandidateReport (résultat de generate_candidate_report)
        model:       Modèle OpenAI à utiliser (default: gpt-4o-mini pour le coût)
        temperature: Créativité du feedback (0.7 = naturel mais pas trop créatif)
        commentaire_correcteur: Commentaire libre du correcteur humain pour ce cas

    Returns:
        PedagogicalFeedback avec texte du commentaire et métadonnées.
    """
    # Vérifier qu'il y a un rapport exploitable
    if report.erreur:
        return PedagogicalFeedback(
            texte=f"Impossible de générer un commentaire pédagogique : {report.erreur}",
            rang_edn_manques=[],
            concepts_cours_cites=[],
            has_critical_miss=False,
            erreur=report.erreur,
        )

    # Construire le contexte
    course_context = _build_course_context(report)
    student_summary = _build_student_summary(report)

    # Identifier les concepts manqués et leurs rangs
    rang_manques = []
    concepts_cites = []
    for vd in report.validant_details:
        entry = get_edn_entry(vd.golden_id)
        if entry:
            concepts_cites.append(vd.golden_name)
            if not vd.found:
                rang_manques.append(entry.rang_edn)

    has_critical = "A" in rang_manques

    # Appel GPT
    correcteur_section = ""
    if commentaire_correcteur and commentaire_correcteur.strip():
        correcteur_section = f"\n\nCOMMENTAIRE DU CORRECTEUR HUMAIN (expert) :\n« {commentaire_correcteur.strip()} »\n(Intègre ce commentaire dans la partie 2 « Votre interprétation » comme conseil expert.)"

    user_message = f"""Voici l'évaluation d'un étudiant sur un cas ECG.

{student_summary}

{course_context}{correcteur_section}

Rédige le commentaire pédagogique en 2 parties (## 1. Référence au cours / ## 2. Votre interprétation)."""

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        feedback_text = response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Erreur GPT pour feedback pédagogique : {e}")
        return PedagogicalFeedback(
            texte=_generate_fallback_feedback(report),
            rang_edn_manques=rang_manques,
            concepts_cours_cites=concepts_cites,
            has_critical_miss=has_critical,
            erreur=str(e)[:200],
        )

    return PedagogicalFeedback(
        texte=feedback_text,
        rang_edn_manques=rang_manques,
        concepts_cours_cites=concepts_cites,
        has_critical_miss=has_critical,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Feedback de secours (sans GPT)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_fallback_feedback(report) -> str:
    """
    Génère un feedback minimal basé uniquement sur la knowledge base,
    sans appel LLM. Utilisé en cas d'erreur GPT.
    """
    parts = []

    # Appréciation
    if report.score_final_pct >= 90:
        parts.append("🎉 Excellente interprétation !")
    elif report.score_final_pct >= 70:
        parts.append("👍 Bonne interprétation, quelques points à préciser.")
    elif report.score_final_pct >= 50:
        parts.append("📚 Interprétation partielle — des éléments importants manquent.")
    else:
        parts.append("💪 Cette interprétation nécessite une révision approfondie.")

    # Rappels de cours pour les validants manqués
    for vd in report.validant_details:
        if not vd.found:
            entry = get_edn_entry(vd.golden_id)
            if entry:
                rang_label = {"A": "indispensable", "B": "important", "C": "complémentaire"}
                parts.append(
                    f"\n❌ {vd.golden_name} (Rang EDN : {entry.rang_edn} — {rang_label.get(entry.rang_edn, '')}) :"
                )
                parts.append(f'📖 « {entry.extrait_cours} » — (Item 231, SFC)')
                if entry.pieges_classiques:
                    for piege in entry.pieges_classiques[:1]:
                        parts.append(f"⚠️ Piège : {piege}")

    # Découvertes
    if report.decouvertes:
        parts.append(f"\n🟢 Vous avez identifié {len(report.decouvertes)} élément(s) pertinent(s) au-delà du barème — c'est un bon signe !")

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Formatage HTML du feedback
# ──────────────────────────────────────────────────────────────────────────────

def format_feedback_html(feedback: PedagogicalFeedback) -> str:
    """Formate le feedback pédagogique en HTML (dark theme), structuré en 3 sections."""
    import re

    # Badge de criticité
    if feedback.has_critical_miss:
        badge = '<span style="background:#F44336; color:white; padding:2px 8px; border-radius:4px; font-size:12px;">⚠️ Concept rang A manqué</span>'
    elif feedback.rang_edn_manques:
        badge = '<span style="background:#FF9800; color:white; padding:2px 8px; border-radius:4px; font-size:12px;">📝 Points à revoir</span>'
    else:
        badge = '<span style="background:#4CAF50; color:white; padding:2px 8px; border-radius:4px; font-size:12px;">✅ Maîtrise confirmée</span>'

    # Extraire les 3 sections
    sections = re.split(r'##\s*\d+\.\s*', feedback.texte)
    section_titles = re.findall(r'##\s*\d+\.\s*(.+)', feedback.texte)

    section_icons = ["📖", "🔍"]
    section_colors = ["#5C6BC0", "#FF9800"]

    sections_html = ""
    if len(section_titles) >= 2 and len(sections) >= 3:
        for idx in range(2):
            title = section_titles[idx].strip()
            content = sections[idx + 1].strip()
            content_html = content.replace("\n\n", "</p><p>").replace("\n", "<br>")
            content_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content_html)
            content_html = re.sub(r'📖\s*«\s*(.+?)\s*»', r'📖 <em style="color:#90CAF9;">«\1»</em>', content_html)
            icon = section_icons[idx]
            color = section_colors[idx]
            sections_html += f"""
        <div style="border-left:4px solid {color}; padding:12px 16px; margin-bottom:12px; background:#1e2d3d; border-radius:4px;">
            <h4 style="color:{color}; margin:0 0 8px 0; font-size:14px;">{icon} {title}</h4>
            <div style="color:#e0e0e0; font-size:13px; line-height:1.6;">
                <p>{content_html}</p>
            </div>
        </div>"""
    else:
        # Fallback
        text_html = feedback.texte.replace("\n\n", "</p><p>").replace("\n", "<br>")
        text_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text_html)
        text_html = re.sub(r'📖\s*«\s*(.+?)\s*»', r'📖 <em style="color:#90CAF9;">«\1»</em>', text_html)
        sections_html = f'<div style="color:#e0e0e0; font-size:14px; line-height:1.7;"><p>{text_html}</p></div>'

    return f"""
    <div style="background:#1a2332; border-radius:8px; padding:16px; margin-top:16px;
                border-left:4px solid #5C6BC0;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h3 style="color:#7986CB; margin:0; font-size:16px;">
                🎓 Commentaire pédagogique — Cours SFC, Item 231
            </h3>
            {badge}
        </div>
        {sections_html}
        <div style="color:#666; font-size:11px; margin-top:12px; border-top:1px solid #333; padding-top:8px;">
            Source : Chapitre 15 — Item 231, Société Française de Cardiologie (SFC), Référentiel CNEC 2e édition
        </div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Ce module nécessite un CandidateReport pour fonctionner.")
    print("Utilisez le notebook candidate_report_demo.ipynb pour tester.")
