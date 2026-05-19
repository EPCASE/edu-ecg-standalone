"""
Exemple 02 — Scoring V3 isolé (sans appel LLM, sans clé OpenAI).

Démontre comment scorer une liste de concepts trouvés vs un golden set,
en utilisant uniquement l'ontologie (couche symbolique).

Usage :
    python examples/02_score_only.py
"""
import sys
import io
from pathlib import Path

# Forcer UTF-8 sur stdout (Windows)
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag_pipeline"))

from scoring_v3 import score_student_response_v3, format_v3_summary, build_negation_map
from semantic_layer import _get_ontology_v2

# Charger l'ontologie V2 (auto-detect du chemin)
_get_ontology_v2()

# Construire la map de négation (utilisée pour convertir "pas de X" → concept positif)
build_negation_map()

# ─── Cas 1 : Match exact ──────────────────────────────────────────────────────
print("\n=== Cas 1 : Match exact ===")
result = score_student_response_v3(
    found_ids=["BLOC_DE_BRANCHE_GAUCHE_COMPLET"],
    expected_ids=["BLOC_DE_BRANCHE_GAUCHE_COMPLET"],
)
print(format_v3_summary(result))

# ─── Cas 2 : Match hiérarchique (parent trouvé pour enfant attendu) ──────────
print("\n=== Cas 2 : Parent trouvé pour enfant attendu (score partiel) ===")
result = score_student_response_v3(
    found_ids=["TACHYCARDIE_ATRIALE"],            # parent
    expected_ids=["FLUTTER_DROIT_TYPIQUE"],       # enfant
)
print(format_v3_summary(result))

# ─── Cas 3 : Concept manqué ───────────────────────────────────────────────────
print("\n=== Cas 3 : Concept manqué ===")
result = score_student_response_v3(
    found_ids=["BRADYCARDIE", "RYTHME_SINUSAL"],
    expected_ids=["BAV_COMPLET"],
)
print(format_v3_summary(result))

# ─── Cas 4 : Avec négation (absent → positif) ─────────────────────────────────
print("\n=== Cas 4 : Négation gérée (absent_ids) ===")
result = score_student_response_v3(
    found_ids=[],
    expected_ids=["ECG_NORMAL"],
    absent_ids=["ANOMALIE_DES_ONDES_T", "BLOC_DE_BRANCHE_DROIT"],  # → suggère ECG normal
)
print(format_v3_summary(result))
