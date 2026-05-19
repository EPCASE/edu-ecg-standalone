"""
🧱 Brique 2 — Extracteur d'Entités Cliniques (NER)
====================================================
Prend le texte libre d'un étudiant et utilise GPT-4o (Structured Outputs)
pour extraire une liste structurée de concepts médicaux bruts.

Principes :
  - Extraction "bête et qualifiée" : ZÉRO normalisation / correction
  - Périmètre ECG strict : ignorer la clinique annexe (âge, douleur…)
  - Chaque entité porte un statut clinique : present / absent / hypothese

L'output est un objet Pydantic `NERExtraction` garanti par l'API OpenAI
via la méthode `.parse()` (Structured Outputs).

Auteur : BMad Team
Date   : 2026-02-25
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schémas Pydantic — strictement respectés par GPT-4o (Structured Outputs)
# ---------------------------------------------------------------------------

class ClinicalEntity(BaseModel):
    """Une entité clinique extraite du texte étudiant."""
    terme_brut: str = Field(
        description="Le concept exact extrait du texte, sans aucune correction orthographique."
    )
    statut: Literal["present", "absent", "hypothese"] = Field(
        description="Le statut clinique du concept."
    )
    contexte_phrase: str = Field(
        description="La phrase complète d'où le terme a été extrait."
    )


class NERExtraction(BaseModel):
    """Résultat complet de l'extraction NER sur un texte étudiant."""
    entites: List[ClinicalEntity]


# ---------------------------------------------------------------------------
# Prompt système — cœur de la Brique 2
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Tu es un expert en lecture d'ECG. Ton rôle est d'extraire toutes les entités cliniques, rythmiques et morphologiques du texte rédigé par un étudiant en médecine.

RÈGLES STRICTES :
1. EXTRACTION PURE : N'essaie JAMAIS de corriger ou normaliser l'orthographe du terme (ex: si l'étudiant écrit "tachi supra", extrais "tachi supra"). 
   -> EXCEPTION UNIQUE : Les valeurs numériques (voir Règle 5).

2. PÉRIMÈTRE ECG — LARGE : Extrais TOUS les termes liés à l'interprétation du tracé ECG. Cela inclut :
   a. Les descriptions morphologiques et rythmiques (ex: "ondes T", "BBD", "tachycardie", "microvoltage", "sous-décalage ST").
   b. Les diagnostics et syndromes affirmés ou suspectés à partir du tracé (ex: "syndrome coronarien", "péricardite", "Brugada").
   c. Les diagnostics ÉTIOLOGIQUES déduits de l'ECG : ce sont les pathologies que l'étudiant conclut en lisant le tracé. Exemples :
      - "hyperkaliémie", "hypokaliémie", "hypercalcémie" (troubles ioniques)
      - "amylose", "hypothermie", "embolie pulmonaire"
      - "intoxication digitalique", "tamponnade"
      - "dysplasie arythmogène du ventricule droit"
   d. Les concepts de stimulation cardiaque (ex: "pacemaker", "stimulation", "pace", "AAI", "DDD").
   -> IGNORE UNIQUEMENT le contexte clinique purement anamnestique du patient (âge, poids, "adressé pour douleur thoracique") qui n'est PAS une conclusion tirée de l'ECG.

3. STATUT CLINIQUE ET NÉGATION : Pour chaque terme, détermine son statut :
   - "present" : le concept est affirmé par l'étudiant.
   - "absent" : le concept est explicitement nié ou écarté.
   - "hypothese" : le concept est suspecté ou incertain (ex: "suspi d'infarctus", "peut-être une amylose").
   
   RÈGLE CRITIQUE SUR LA NÉGATION :
   Quand l'étudiant écrit "pas de X", "sans X", "absence de X", "pas d'X", "ni X", "élimine X",
   tu dois IMPÉRATIVEMENT :
     a) Mettre le statut à "absent"
     b) Extraire UNIQUEMENT le concept X dans terme_brut, SANS la négation
   
   EXEMPLES OBLIGATOIRES :
     - "Pas de trouble de repolarisation"   → terme_brut="trouble de repolarisation", statut="absent"
     - "sans BBD"                           → terme_brut="BBD",                       statut="absent"
     - "pas de séquelle de nécrose"         → terme_brut="séquelle de nécrose",       statut="absent"
     - "pas d'HVG"                          → terme_brut="HVG",                       statut="absent"
     - "pas de critère de Sgarbossa"        → terme_brut="critère de Sgarbossa",      statut="absent"
     - "Pas de signe d'embolie pulmonaire"  → terme_brut="embolie pulmonaire",        statut="absent"
     - "ni FA ni flutter"                   → 2 entités : terme_brut="FA" statut="absent" + terme_brut="flutter" statut="absent"
   
   ERREUR TYPIQUE À ÉVITER :
     ❌ terme_brut="Pas de trouble de repolarisation", statut="present"   ← FAUX !
     ✅ terme_brut="trouble de repolarisation",        statut="absent"    ← CORRECT
   
4. GESTION DES ADJECTIFS ET MODIFICATEURS (MÉTHODE LEGO) : 
   Tu dois distinguer deux situations pour le groupement sémantique :
   a) Les ondes et segments simples : Garde l'adjectif attaché à l'onde (ex: extrais l'entité complète "Onde T symétrique", "PR long", "Sus-décalage V1").
   b) Les troubles du rythme et extrasystoles : Tu dois IMPÉRATIVEMENT séparer le diagnostic principal de ses adjectifs descriptifs (durée, séquence, morphologie globale) pour en faire des entités cliniques distinctes.
      -> Exemples de modificateurs à extraire SEULS : "monomorphe", "polymorphe", "soutenu", "non soutenu", "en salve", "isolée", "bigéminée", "trigéminée".
      -> EXEMPLE D'APPLICATION : Si l'étudiant écrit "Salves non soutenues d'ESV polymorphes", tu dois extraire 4 entités distinctes : {"terme_brut": "Salve"}, {"terme_brut": "non soutenues"}, {"terme_brut": "ESV"}, {"terme_brut": "polymorphes"}. Ne crée JAMAIS de terme fusionné comme "ESV polymorphe".
    -> exception, pour les modificateurs de durée (ex: "BBD complet", "BAV complet", "BAV 2 Mobitz 2") qui peuvent être extraits en même temps que le diagnostic principal, car ils font partie intégrante du concept clinique.
    
5. TRADUCTION CLINIQUE DES MESURES : Les espaces de recherche ne comprennent pas les chiffres. Si l'étudiant donne une valeur numérique brute, ne l'extrais JAMAIS telle quelle dans `terme_brut`. Tu dois la traduire en conclusion clinique standardisée, tout en gardant la valeur d'origine dans `contexte_phrase`.
Applique STRICTEMENT ces règles de conversion pour générer le `terme_brut` :
   - Fréquence (bpm, /min) : <60 -> "Bradycardie", 60-100 -> "Normocarde", >100 -> "Tachycardie"
   - QRS : <120 ms -> "QRS fins", >=120 ms -> "QRS large"
   - PR : <120 ms -> "PR court", 120-200 ms -> "PR normal", >200 ms -> "PR allongé"
   - Axe (degrés) : entre -30 et 90 -> "Axe normal", <-30 -> "Déviation axiale gauche", >90 -> "Déviation axiale droite"

6. EXPANSION DES ABRÉVIATIONS ECG : Quand l'étudiant utilise une abréviation ECG standard et connue, tu DOIS l'expanser dans `terme_brut` en forme longue française. L'objectif est que les espaces de recherche puissent retrouver le concept.
   Applique STRICTEMENT ces expansions (liste non exhaustive) :
     - "RS" ou "rs" → "Rythme sinusal"
     - "FA" ou "fa" → "Fibrillation atriale"
     - "HBAG" → "Hémibloc antérieur gauche"
     - "HBPG" → "Hémibloc postérieur gauche"
     - "ESV" → "Extrasystole ventriculaire"
     - "ESA" ou "ESSV" → "Extrasystole supraventriculaire"
     - "TV" → "Tachycardie ventriculaire"
     - "TSV" → "Tachycardie supraventriculaire"
     - "FV" → "Fibrillation ventriculaire"
     - "BAV" ou "BBG" ou "BBD" → pas besoin de le faire car c'est déjà une abréviation très courante
     - "IDM" → "Infarctus du myocarde"
     - "SCA" → "Syndrome coronarien aigu"
     - "HVG" → "Hypertrophie ventriculaire gauche"
     - "HVD" → "Hypertrophie ventriculaire droite"
     - "HAG" → "Hypertrophie atriale gauche"
     - "HAD" → "Hypertrophie atriale droite"
     - "WPW" → "Wolff-Parkinson-White"
     - "EP" ou "embolie pulmonaire" (garder tel quel si déjà en forme longue)
   Si l'abréviation N'EST PAS dans cette liste et que tu n'es pas sûr de sa signification ECG, garde-la telle quelle.
   
   ATTENTION : cette règle NE S'APPLIQUE PAS aux noms d'ondes/segments (P, QRS, ST, T, U, R, S, Q) ni aux dérivations (V1-V6, D1-D3, aVR, aVL, aVF).

7. REFORMULATION MORPHOLOGIQUE LÉGÈRE : Si l'étudiant utilise une forme adjectivale ou passive d'un concept ECG connu, tu peux reformuler en forme nominale standard dans `terme_brut`, à condition que le sens soit strictement identique.
   Exemples :
     - "onde R rabotée en antérieur" → "Rabotage de l'onde R en antérieur"
     - "onde R rabottée" → "Rabotage de l'onde R"
     - "QT allongé" → "Allongement du QT" (ou garder "QT allongé", les deux sont acceptés)
   
   NE JAMAIS inventer un concept qui n'existe pas. En cas de doute, garder la formulation de l'étudiant.
""".strip()

# Modèle OpenAI compatible Structured Outputs
MODEL = "gpt-4o-2024-08-06"


# ---------------------------------------------------------------------------
# Client OpenAI (singleton module-level, initialisé à la demande)
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Retourne le client OpenAI, en le créant si nécessaire."""
    global _client
    if _client is None:
        # Chercher le .env dans plusieurs emplacements possibles
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
            load_dotenv()  # fallback

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY non trouvée. "
                "Ajoutez-la dans un fichier .env ou en variable d'environnement."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Fonction principale — Brique 2
# ---------------------------------------------------------------------------

def extract_clinical_terms(texte_etudiant: str) -> NERExtraction:
    """
    Extrait les entités cliniques ECG d'un texte étudiant via GPT-4o.

    Args:
        texte_etudiant: Le texte libre rédigé par l'étudiant.

    Returns:
        NERExtraction: Objet Pydantic contenant la liste des entités extraites,
                       chacune avec terme_brut, statut et contexte_phrase.

    Raises:
        RuntimeError: Si la clé API est manquante.
        openai.APIError: Si l'appel API échoue.
    """
    client = _get_client()

    logger.info(f"🔬 NER Extraction — texte de {len(texte_etudiant)} caractères")

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Texte de l'étudiant : {texte_etudiant}"},
        ],
        response_format=NERExtraction,
    )

    result = response.choices[0].message.parsed

    logger.info(f"✅ {len(result.entites)} entités extraites")
    for ent in result.entites:
        logger.debug(f"   [{ent.statut:>10}] {ent.terme_brut}")

    return result


# ---------------------------------------------------------------------------
# Point d'entrée CLI pour test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Texte de test (spécification)
    texte_test = (
        "Patient de 54 ans avec douleur thoracique. "
        "Rythme sinusal. Pas de BBD visible. "
        "On suspecte une amylose devant le microvoltage."
    )

    print(f"\n📝 Texte étudiant :\n   {texte_test}\n")
    print("=" * 60)

    result = extract_clinical_terms(texte_test)

    print(f"\n🔬 {len(result.entites)} entités extraites :\n")
    for i, ent in enumerate(result.entites, 1):
        print(f"  {i}. [{ent.statut:>10}] \"{ent.terme_brut}\"")
        print(f"     └─ Contexte : \"{ent.contexte_phrase}\"\n")

    print("✅ Brique 2 — Extraction NER terminée.")
