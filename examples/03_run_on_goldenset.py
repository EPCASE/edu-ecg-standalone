"""
Exemple 03 — Re-jouer le pipeline V3 sur un vrai cas du goldenset
avec une vraie réponse étudiante issue de testing_data/corrections/.

Usage :
    python examples/03_run_on_goldenset.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag_pipeline"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from candidate_report import generate_candidate_report, format_report_text

DATA = ROOT / "testing_data"
GOLDEN_DIR = DATA / "goldenset"
CORR_DIR = DATA / "corrections"


def load_goldenset_index() -> dict:
    """Mappe name (ex: '2') → metadata complète."""
    index = {}
    for case_folder in sorted(GOLDEN_DIR.iterdir()):
        meta_file = case_folder / "metadata.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            index[meta.get("name", case_folder.name)] = meta
    return index


def find_student_with_answer(case_name: str) -> tuple[str, str, dict] | None:
    """Trouve un étudiant qui a répondu au cas `case_name` (non vide)."""
    for student_file in sorted(CORR_DIR.glob("ECG-*.json")):
        data = json.loads(student_file.read_text(encoding="utf-8"))
        case = data.get("cases", {}).get(case_name, {})
        text = (case.get("student_text") or "").strip()
        if text:
            return student_file.stem, text, case.get("report") or {}
    return None


def main():
    # ── Choisir un cas (ici : cas 2 = "BAV complet") ──────────────────────────
    case_name = "2"
    index = load_goldenset_index()
    if case_name not in index:
        print(f"❌ Cas '{case_name}' introuvable. Disponibles : {sorted(index.keys())}")
        return

    meta = index[case_name]
    print("=" * 72)
    print(f"🩺 Cas {case_name} — {meta.get('category', '?')} "
          f"({meta.get('difficulty', '?')})")
    print(f"   Diagnostic attendu : {meta.get('diagnostic_principal', '?')}")
    print(f"   Concepts attendus  : {meta.get('expected_concepts', [])}")
    print("=" * 72)

    # ── Trouver un étudiant qui a répondu ─────────────────────────────────────
    found = find_student_with_answer(case_name)
    if not found:
        print(f"⚠️  Aucune réponse étudiante trouvée pour le cas {case_name}.")
        return

    student_code, student_text, original_report = found
    print(f"\n👤 Étudiant : {student_code}")
    print(f"📝 Réponse :\n   {student_text}\n")
    print(f"📊 Score original (v1.0) : {original_report.get('score_final_pct', '?')}%")

    # ── Re-jouer le pipeline V3 ───────────────────────────────────────────────
    print("\n⏳ Re-jouer le pipeline V3 (NER → Search → Juge → Scoring V3)…\n")

    # Construire les golden_names + golden_roles à partir des annotations
    annotations = meta.get("annotations", [])
    golden_names = [a["concept"] for a in annotations]
    golden_roles = [
        "validant" if "validant" in a.get("annotation_role", "").lower() else "descripteur"
        for a in annotations
    ]

    report = generate_candidate_report(
        texte_etudiant=student_text,
        golden_names=golden_names,
        golden_roles=golden_roles,
        diagnostic_principal=meta.get("diagnostic_principal", ""),
        with_feedback=False,
    )

    print(format_report_text(report))
    print("\n" + "=" * 72)
    print(f"⏱️  Latence V3  : {report.latence_s:.2f}s")
    print(f"📊 Score V3   : {report.score_final_pct}%")
    print(f"📊 Score v1.0 : {original_report.get('score_final_pct', '?')}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
