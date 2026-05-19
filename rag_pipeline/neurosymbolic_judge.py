"""
🧱 Brique 4 — Le Juge Neurosymbolique (Coupe-Circuit + GPT-4o-mini)
=====================================================================
Prend un terme brut + son contexte + les Top-K candidats de la Brique 3,
et retourne l'ontology_id final ou "NONE".

Pipeline en 2 étapes :
  1. Coupe-Circuit : si le candidat n°1 a is_exact_match=True,
     on retourne immédiatement son ontology_id (bypass LLM).
  2. Juge LLM (QCM) : sinon, on soumet le Top-K à GPT-4o-mini
     sous forme de QCM. Le LLM peut répondre NONE si aucun candidat
     ne correspond cliniquement.

Dépendances :
  - HybridSearchEngine (Brique 3) pour le flag is_exact_match
  - normalize_text (Brique 1) pour la normalisation
  - OpenAI API (GPT-4o-mini) pour le fallback LLM

Auteur : BMad Team
Date   : 2026-02-25
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Import de la normalisation Brique 1
from ontology_index import normalize_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schéma Pydantic — Structured Outputs pour le Juge LLM
# ---------------------------------------------------------------------------

class ConceptMatching(BaseModel):
    """Réponse structurée du Juge LLM."""
    id_ontologie: str = Field(
        description=(
            "L'ontology_id du concept retenu parmi les candidats proposés "
            "(ex: 'FIBRILLATION_ATRIALE'), ou 'NONE' si aucun ne correspond."
        )
    )
    justification: str = Field(
        description=(
            "Explication courte du choix : pourquoi ce concept correspond "
            "au terme de l'étudiant, ou pourquoi aucun ne correspond."
        )
    )
    confiance: int = Field(
        default=50,
        description=(
            "Niveau de confiance de 0 à 100 dans le choix effectué. "
            "100 = certitude absolue, 0 = pure conjecture."
        )
    )


# ---------------------------------------------------------------------------
# Prompt système — Le Juge
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Tu es un expert en cardiologie et en lecture d'ECG. Ton rôle est de relier un terme clinique brut écrit par un étudiant au concept officiel correct de notre ontologie.

RÈGLES STRICTES (par ordre de priorité) :

1. Analyse le terme de l'étudiant ET son contexte (la phrase d'où il provient).

2. Choisis le concept correspondant PARMI LES OPTIONS FOURNIES UNIQUEMENT.

3. CORRESPONDANCE EXACTE D'ABORD : si le terme de l'étudiant correspond exactement (mot pour mot) à un concept, choisis CE concept. Ne "monte pas en spécificité" artificiellement.
   Exemples CRITIQUES :
   - "Tachycardie" SEUL (sans qualificatif) → TACHYCARDIE (descripteur de fréquence), PAS TACHYCARDIE_VENTRICULAIRE
   - "QRS large" ou "QRS larges" → QRS_LARGE (descripteur morphologique), PAS BLOC_INTRAVENTRICULAIRE
   - "Échappement" SEUL → ECHAPPEMENT, PAS ECHAPPEMENT_VENTRICULAIRE (sauf si le contexte précise "ventriculaire")
   La spécificité s'applique UNIQUEMENT quand le terme de l'étudiant contient LUI-MÊME des précisions supplémentaires.

4. SPÉCIFICITÉ MAXIMALE (quand le terme est qualifié) : si le terme de l'étudiant contient des précisions, choisis le concept le plus spécifique qui correspond à ces précisions.
   Exemples :
   - "Flutter typique" → FLUTTER_DROIT_TYPIQUE (pas FLUTTER_ATRIAL qui est le parent générique)
   - "Bloc de branche gauche complet" → BLOC_DE_BRANCHE_GAUCHE_COMPLET (pas BLOC_DE_BRANCHE_GAUCHE)
   - "Mobitz 2" → BAV_2_MOBITZ_2 (pas BAV qui est trop vague)
   - "Tachycardie ventriculaire" → TACHYCARDIE_VENTRICULAIRE (le terme contient "ventriculaire")

5. DIAGNOSTIC > DESCRIPTEUR : si un candidat est un diagnostic (catégorie DIAGNOSTIC_URGENT ou DIAGNOSTIC_MAJEUR) et un autre est un simple descripteur (DESCRIPTEUR_ECG) ou un territoire, préfère le diagnostic UNIQUEMENT quand le terme de l'étudiant désigne clairement cette pathologie ou ce syndrome.
   Exemples :
   - "Voie accessoire" → FAISCEAU_ACCESSOIRE_À_CONDUCTION_ANTÉROGRADE (diagnostic), pas un territoire anatomique
   - "Hyperkaliémie" → HYPERKALIÉMIE (diagnostic), pas un signe ECG isolé
   Contre-exemples :
   - "Tachycardie" (terme isolé dans un contexte de FA) → TACHYCARDIE (descripteur), PAS TACHYCARDIE_VENTRICULAIRE

6. NE PAS GÉNÉRALISER : un terme qui décrit un élément spécifique (axe, onde, segment, intervalle, fréquence) ne doit JAMAIS être mappé vers un concept global comme ECG_NORMAL ou ECG_PAS_NORMAL. Choisis le concept spécifique qui décrit cet élément.
   Exemples CRITIQUES :
   - "axe gauche physiologique" → AXE_NORMAL_DU_QRS (pas ECG_NORMAL)
   - "onde P normale" → ONDE_P_NORMALE (pas ECG_NORMAL)
   - "QRS fins" → QRS_FINS (pas QRS_NORMAL ou ECG_NORMAL)
   - "rythme régulier" → RYTHME_REGULIER (pas ECG_NORMAL)

7. Si le terme de l'étudiant désigne la même pathologie, morphologie ou rythme que l'une des options, renvoie son ontology_id exact.

8. Si AUCUNE option ne correspond cliniquement (le terme parle d'autre chose, ou les options sont toutes fausses), renvoie impérativement 'NONE'.

9. Ne devine pas : si tu n'es pas sûr, renvoie 'NONE'.
""".strip()

# Modèle rapide et économique pour le QCM
MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Client OpenAI (singleton module-level)
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Retourne le client OpenAI, en le créant si nécessaire."""
    global _client
    if _client is None:
        env_candidates = [
            Path(".env"),
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / "ECG lecture" / ".env",
        ]
        for env_path in env_candidates:
            if env_path.exists():
                load_dotenv(env_path)
                break
        else:
            load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY non trouvée. "
                "Ajoutez-la dans un fichier .env ou en variable d'environnement."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Helper — Résumé compact des candidats pour stockage
# ---------------------------------------------------------------------------

def _extract_candidats_resume(top_k_candidates: List[Dict]) -> List[Dict]:
    """
    Extrait un résumé compact des Top-K candidats pour stockage dans le rapport.
    
    On garde uniquement les champs utiles à l'analyse (pas les métadonnées
    lourdes comme source_type ou categorie qui sont dans l'ontologie).
    """
    resume = []
    for c in top_k_candidates:
        resume.append({
            "ontology_id": c["ontology_id"],
            "concept_name": c["concept_name"],
            "surface_form": c["surface_form"],
            "rrf_score": c.get("rrf_score", 0.0),
            "cosine_score": c.get("cosine_score", 0.0),
            "bm25_score": c.get("bm25_score", 0.0),
            "is_exact_match": c.get("is_exact_match", False),
        })
    return resume


# ---------------------------------------------------------------------------
# Fonction principale — Brique 4
# ---------------------------------------------------------------------------

def resolve_term_to_ontology(
    terme_brut: str,
    contexte_phrase: str,
    top_k_candidates: List[Dict],
) -> Dict:
    """
    Résout un terme brut extrait par GPT-4o vers un ontology_id officiel.

    Pipeline :
      1. Si pas de candidats → NONE
      2. Coupe-circuit : si candidat n°1 a is_exact_match=True → bypass LLM
      3. Sinon : Juge LLM (QCM GPT-4o-mini) → ontology_id ou NONE

    Args:
        terme_brut:       Le terme brut de l'étudiant (ex: "tachi supra").
        contexte_phrase:  La phrase d'origine (ex: "On note une tachi supra").
        top_k_candidates: Liste de dicts issus de HybridSearchEngine.search_top_k().

    Returns:
        Dict contenant :
          - ontology_id    : str — l'ID retenu ou "NONE"
          - concept_name   : str — le nom canonique (ou "" si NONE)
          - method         : str — "coupe_circuit" ou "juge_llm" ou "no_candidates"
          - justification  : str — explication du choix
          - candidats_soumis: int — nombre de candidats soumis au juge
          - top_k_candidats: list — les Top-K candidats avec scores (rrf, cosine, bm25)
          - llm_confiance  : int — confiance auto-évaluée par le LLM (0-100), -1 si coupe-circuit
    """
    # --- Cas trivial : pas de candidats ---
    if not top_k_candidates:
        logger.info(f"⚠️  Pas de candidats pour : '{terme_brut}' → NONE")
        return {
            "ontology_id": "NONE",
            "concept_name": "",
            "method": "no_candidates",
            "justification": "Aucun candidat retourné par la recherche hybride.",
            "candidats_soumis": 0,
            "top_k_candidats": [],
            "llm_confiance": -1,
        }

    # --- Étape 1 : Coupe-circuit (exact match = PRIORITAIRE) ---
    # Si le terme brut de l'étudiant matche exactement une surface_form
    # de l'ontologie → c'est ce concept, sans exception.
    # Philosophie : l'étudiant a écrit ce qu'il voulait dire.
    # "Tachycardie" = TACHYCARDIE (descripteur de fréquence).
    # S'il voulait dire "Tachycardie ventriculaire", il l'aurait écrit.
    # C'est au scoring (Brique 5) de relier TACHYCARDIE aux patterns
    # via les relations requires/supports de l'ontologie.
    candidat_1 = top_k_candidates[0]
    if candidat_1.get("is_exact_match", False):
        logger.info(
            f"⚡ Coupe-circuit : '{terme_brut}' → "
            f"{candidat_1['ontology_id']} (\"{candidat_1['surface_form']}\")"
        )
        candidats_resume = _extract_candidats_resume(top_k_candidates)
        return {
            "ontology_id": candidat_1["ontology_id"],
            "concept_name": candidat_1["concept_name"],
            "method": "coupe_circuit",
            "justification": (
                f"Match exact normalisé : "
                f"'{normalize_text(terme_brut)}' == surface_form du concept."
            ),
            "candidats_soumis": 0,
            "top_k_candidats": candidats_resume,
            "llm_confiance": -1,  # Pas de LLM, match déterministe
        }

    # --- Étape 2 : Juge LLM (QCM) ---
    juge_result = _juge_llm(terme_brut, contexte_phrase, top_k_candidates)

    # --- Étape 3 : Fallback sous-termes si le Juge renvoie NONE ---
    # Quand un terme composé comme "ESV infundibulaire droite postéroseptale"
    # échoue, on tente chaque sous-terme individuellement pour récupérer
    # le concept principal (ex: "ESV" → EXTRASYSTOLE_VENTRICULAIRE).
    if juge_result["ontology_id"] == "NONE":
        subterm_result = _fallback_subtokens(terme_brut, contexte_phrase)
        if subterm_result is not None:
            # Enrichir le fallback avec les candidats du terme original
            subterm_result["top_k_candidats"] = _extract_candidats_resume(top_k_candidates)
            subterm_result["llm_confiance"] = juge_result.get("llm_confiance", -1)
            return subterm_result

    # Enrichir le résultat du juge avec les candidats
    juge_result["top_k_candidats"] = _extract_candidats_resume(top_k_candidates)
    return juge_result


def _fallback_subtokens(
    terme_brut: str,
    contexte_phrase: str,
) -> Optional[Dict]:
    """
    Fallback : quand le Juge renvoie NONE sur un terme composé,
    on isole chaque sous-terme et retente un Search + coupe-circuit.

    Seuls les matchs **exacts** (is_exact_match=True) sont acceptés.
    On privilégie le sous-terme avec le poids clinique le plus élevé.

    Ex: "ESV infundibulaire droite postéroseptale"
        → sous-termes: ["ESV", "infundibulaire", "droite", "postéroseptale"]
        → "ESV" exact-match EXTRASYSTOLE_VENTRICULAIRE (poids=2) ✅

    Returns:
        Dict résolution si un sous-terme matche, ou None.
    """
    # Importer ici pour éviter circular import au module-level
    from hybrid_search import HybridSearchEngine

    words = terme_brut.split()
    if len(words) <= 1:
        return None  # Terme simple, pas de sous-termes à essayer

    # Lazy-load du moteur (on réutilise un singleton si possible)
    if not hasattr(_fallback_subtokens, "_engine"):
        try:
            index_dir = str(Path(__file__).parent / "rag_index")
            _fallback_subtokens._engine = HybridSearchEngine(index_dir)
        except Exception:
            return None

    engine = _fallback_subtokens._engine
    best_match = None
    best_poids = -1

    for word in words:
        if len(word) <= 1:
            continue
        candidates = engine.search_top_k(word, k=3)
        if not candidates:
            continue
        c1 = candidates[0]
        if c1.get("is_exact_match", False):
            poids = c1.get("poids", 1)
            if poids > best_poids:
                best_poids = poids
                best_match = c1

    if best_match is not None:
        logger.info(
            f"🔄 Fallback sous-terme : '{terme_brut}' → "
            f"{best_match['ontology_id']} (via sous-terme exact, poids={best_poids})"
        )
        return {
            "ontology_id": best_match["ontology_id"],
            "concept_name": best_match["concept_name"],
            "method": "fallback_subterm",
            "justification": (
                f"Terme composé '{terme_brut}' non résolu par le Juge. "
                f"Sous-terme '{best_match['surface_form']}' matche exactement "
                f"{best_match['ontology_id']}."
            ),
            "candidats_soumis": 0,
        }

    return None


def _juge_llm(
    terme_brut: str,
    contexte_phrase: str,
    candidates: List[Dict],
) -> Dict:
    """
    Soumet le terme + candidats au Juge GPT-4o-mini sous forme de QCM.

    Returns:
        Dict avec ontology_id, concept_name, method, justification.
    """
    client = _get_client()

    # Préparer les options du QCM
    options_lines = []
    valid_ids = set()
    for c in candidates:
        oid = c["ontology_id"]
        valid_ids.add(oid)
        poids = c.get("poids", "?")
        options_lines.append(
            f"- {c['concept_name']} (ID: {oid}) "
            f"[catégorie: {c['categorie']}, poids: {poids}, "
            f"surface matchée: \"{c['surface_form']}\"]"
        )
    options_text = "\n".join(options_lines)

    prompt_user = (
        f'Terme de l\'étudiant : "{terme_brut}"\n'
        f'Phrase d\'origine : "{contexte_phrase}"\n\n'
        f"Candidats de l'ontologie proposés :\n{options_text}\n\n"
        f"Rappel : renvoie l'ontology_id exact d'un candidat ci-dessus, "
        f"ou 'NONE' si aucun ne correspond."
    )

    logger.info(
        f"🧑‍⚖️ Juge LLM : '{terme_brut}' — "
        f"{len(candidates)} candidats soumis"
    )

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user},
        ],
        response_format=ConceptMatching,
    )

    result = response.choices[0].message.parsed
    chosen_id = result.id_ontologie.strip().upper()  # .upper() car GPT-4o-mini renvoie parfois du MixedCase
    justification = result.justification.strip()
    confiance = result.confiance

    # Validation : le LLM doit renvoyer un ID valide ou "NONE"
    if chosen_id != "NONE" and chosen_id not in valid_ids:
        logger.warning(
            f"⚠️  Le Juge a renvoyé un ID invalide : '{chosen_id}'. "
            f"IDs valides : {valid_ids}. Forçage → NONE."
        )
        chosen_id = "NONE"
        justification = (
            f"ID invalide renvoyé par le LLM ('{result.id_ontologie}'). "
            f"Forçage NONE."
        )

    # Récupérer le concept_name si match
    concept_name = ""
    if chosen_id != "NONE":
        for c in candidates:
            if c["ontology_id"] == chosen_id:
                concept_name = c["concept_name"]
                break

    logger.info(
        f"{'✅' if chosen_id != 'NONE' else '❌'} Juge LLM : "
        f"'{terme_brut}' → {chosen_id}"
        f"{f' ({concept_name})' if concept_name else ''}"
    )

    return {
        "ontology_id": chosen_id,
        "concept_name": concept_name,
        "method": "juge_llm",
        "justification": justification,
        "candidats_soumis": len(candidates),
        "llm_confiance": confiance,
    }


# ---------------------------------------------------------------------------
# Point d'entrée CLI pour test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    sys.path.insert(0, str(Path(__file__).parent))
    from hybrid_search import HybridSearchEngine

    index_dir = str(Path(__file__).parent / "rag_index")
    engine = HybridSearchEngine(index_dir)

    # Cas de test
    test_cases = [
        # (terme_brut, contexte_phrase)
        ("FA", "On note une FA rapide."),
        ("BBG", "BBG complet avec QRS larges."),
        ("tachi supra", "On note une tachi supra."),
        ("fibrillation auriculaire", "Fibrillation auriculaire rapide à 150/min."),
        ("truc bizarre", "Il y a un truc bizarre sur l'ECG."),
    ]

    print("\n🧑‍⚖️ BRIQUE 4 — Juge Neurosymbolique")
    print("=" * 60)

    for terme, contexte in test_cases:
        print(f"\n🔍 Terme : \"{terme}\"")
        print(f"   Contexte : \"{contexte}\"")

        candidates = engine.search_top_k(terme, k=5)
        result = resolve_term_to_ontology(terme, contexte, candidates)

        print(f"   → {result['ontology_id']} ({result['method']})")
        print(f"   💬 {result['justification']}")

    print("\n✅ Brique 4 — Juge Neurosymbolique terminé.")
