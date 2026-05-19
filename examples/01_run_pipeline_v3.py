"""
Exemple 01 — Pipeline V3 complet sur une réponse étudiante.

Usage :
    python examples/01_run_pipeline_v3.py

Pré-requis :
    - pip install -r requirements.txt
    - Copier .env.example en .env et y mettre votre OPENAI_API_KEY
"""
import sys
import io
from pathlib import Path

# Forcer UTF-8 sur stdout (Windows)
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Ajouter rag_pipeline/ au PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag_pipeline"))

# Charger la clé OpenAI depuis .env
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from candidate_report import generate_candidate_report, format_report_text


# ─── Données d'exemple ────────────────────────────────────────────────────────
# Un cas de Bloc de Branche Gauche complet (BBG) : un étudiant l'a bien identifié

TEXTE_ETUDIANT = (
    "Rythme sinusal régulier à 75 bpm. QRS large à 140 ms, "
    "aspect en M en V5-V6 et en QS en V1-V2. "
    "Anomalies secondaires de la repolarisation. "
    "Conclusion : Bloc de branche gauche complet."
)

GOLDEN_NAMES = ["Bloc de branche gauche complet"]
GOLDEN_IDS = ["BLOC_DE_BRANCHE_GAUCHE_COMPLET"]
GOLDEN_ROLES = ["validant"]
DIAGNOSTIC_PRINCIPAL = "Bloc de branche gauche complet"


# ─── Exécution ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("🩺 Pipeline RAG neurosymbolique V3 — Exemple BBG complet")
    print("=" * 70)
    print(f"\n📝 Texte étudiant :\n{TEXTE_ETUDIANT}\n")
    print(f"🎯 Golden set : {GOLDEN_NAMES}\n")
    print("⏳ Exécution du pipeline (NER → Search → Juge → Scoring V3 → Feedback)…\n")

    report = generate_candidate_report(
        texte_etudiant=TEXTE_ETUDIANT,
        golden_names=GOLDEN_NAMES,
        golden_ids=GOLDEN_IDS,
        golden_roles=GOLDEN_ROLES,
        diagnostic_principal=DIAGNOSTIC_PRINCIPAL,
        with_feedback=False,  # passer à True pour le feedback pédagogique GPT
    )

    print(format_report_text(report))
    print("\n" + "=" * 70)
    print(f"⏱️  Latence totale : {report.latence_s:.2f}s")
    print(f"📊 Score final : {report.score_final_pct}%")
    print("=" * 70)
