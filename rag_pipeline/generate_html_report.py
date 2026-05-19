"""
📄 Générateur de rapport HTML — Correction de tous les étudiants × 15 cas ECG
================================================================================
Lit le CSV collecteur, exécute le pipeline RAG sur chaque réponse, et produit
un fichier HTML autonome (ouvrable dans un navigateur, imprimable PDF via Ctrl+P).

Usage :
    python generate_html_report.py                       # tous les étudiants
    python generate_html_report.py --students ECG-WY55   # un seul étudiant
    python generate_html_report.py --students ECG-WY55 ECG-7512  # plusieurs
    python generate_html_report.py --no-feedback         # sans feedback GPT (plus rapide)
    python generate_html_report.py --web                 # mode web : images séparées (léger)

Le fichier HTML est écrit dans :  exports/rapport_corrections_YYYY-MM-DD.html
En mode --web :                   exports/site/index.html + images/*.png (~5 Mo total)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = Path(r"C:\Users\Administrateur\bmad\ECG lecture")
EVAL_ROOT    = Path(r"C:\Users\Administrateur\bmad\ECG evaluation")
RAG_ROOT     = SCRIPT_DIR
IMAGES_ROOT  = Path(r"C:\Users\Administrateur\ECG collector\images")
STUDENTS_DIR = Path(r"C:\Users\Administrateur\ECG collector\corrections\students")
EXPORTS_DIR  = RAG_ROOT / "exports"

CSV_DEFAULT  = EVAL_ROOT / "ECG_Collector_Data - responses(1).csv"

# Mode web : images comme fichiers séparés au lieu de base64
WEB_MODE = False

# ── Setup ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(RAG_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
os.chdir(str(RAG_ROOT))

from candidate_report import (
    generate_candidate_report,
    format_report_html,
    CandidateReport,
)
from scoring_v3 import find_owl_concept

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

def load_golden_set() -> Dict[int, dict]:
    """Charge les 15 cas du golden set depuis les metadata.json."""
    golden = {}
    goldenset_dir = EVAL_ROOT / "goldenset"

    for case_dir in sorted(goldenset_dir.iterdir()):
        meta_file = case_dir / "metadata.json"
        if not (case_dir.is_dir() and meta_file.exists()):
            continue

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        case_num = int(meta["name"])
        names, ids, roles = [], [], []
        for ann in meta.get("annotations", []):
            owl = find_owl_concept(ann["concept"])
            if owl:
                names.append(ann["concept"])
                ids.append(owl["ontology_id"])
                role = "validant" if "validant" in ann.get("annotation_role", "").lower() else "descripteur"
                roles.append(role)

        golden[case_num] = {
            "diagnostic_principal": meta.get("diagnostic_principal", ""),
            "category": meta.get("category", ""),
            "golden_names": names,
            "golden_ids": ids,
            "golden_roles": roles,
            "annotations": meta.get("annotations", []),
            "commentaire_correcteur": meta.get("commentaire_correcteur", ""),
        }

    return golden


def load_student_json(students_dir: Path = STUDENTS_DIR) -> Dict[str, Dict[int, str]]:
    """
    Lit les JSON étudiants (ECG-*.json) et retourne {code: {cas_num: texte}}.
    Ne charge que les student_text, PAS les anciens reports.
    """
    students: Dict[str, Dict[int, str]] = {}

    for fpath in sorted(students_dir.glob("ECG-*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        code = data["code"]
        responses: Dict[int, str] = {}
        for cas_num_str, cas_data in data.get("cases", {}).items():
            txt = cas_data.get("student_text", "").strip()
            if txt:
                responses[int(cas_num_str)] = txt
        students[code] = responses

    return students


def load_student_csv(csv_path: Path) -> Dict[str, Dict[int, str]]:
    """
    Lit le CSV collecteur et retourne {code_etudiant: {cas_num: texte}}.
    Gère les champs multilignes (entre guillemets CSV).
    """
    students: Dict[str, Dict[int, str]] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["code"].strip()
            responses: Dict[int, str] = {}
            for i in range(1, 16):
                col = f"cas_{i:02d}"
                txt = row.get(col, "").strip()
                if txt:
                    # Normaliser les retours à la ligne
                    txt = " ".join(txt.replace("\r\n", " ").replace("\n", " ").split())
                    responses[i] = txt
            students[code] = responses

    return students


# ══════════════════════════════════════════════════════════════════════════════
# 2. Exécution du pipeline
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_student(
    code: str,
    responses: Dict[int, str],
    golden: Dict[int, dict],
    with_feedback: bool = False,
) -> List[Tuple[int, str, Optional[CandidateReport]]]:
    """
    Évalue toutes les réponses d'un étudiant.
    Retourne [(cas_num, texte, report ou None si vide)].
    """
    results = []
    for cas_num in range(1, 16):
        texte = responses.get(cas_num)
        if not texte or cas_num not in golden:
            results.append((cas_num, texte or "", None))
            continue

        cas = golden[cas_num]
        try:
            report = generate_candidate_report(
                texte_etudiant=texte,
                golden_names=cas["golden_names"],
                golden_ids=cas["golden_ids"],
                golden_roles=cas["golden_roles"],
                diagnostic_principal=cas["diagnostic_principal"],
                with_feedback=with_feedback,
                commentaire_correcteur=cas.get("commentaire_correcteur", ""),
            )
            results.append((cas_num, texte, report))
        except Exception as e:
            logger.error(f"  ❌ Erreur cas {cas_num} pour {code}: {e}")
            results.append((cas_num, texte, None))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. Génération HTML
# ══════════════════════════════════════════════════════════════════════════════

def _ecg_image_tag(cas_num: int) -> str:
    """Génère la balise img pour un ECG (base64 ou chemin relatif en mode web)."""
    img_path = IMAGES_ROOT / f"{cas_num}.png"
    if not img_path.exists():
        return f'<div style="color:#888;">📸 ECG cas {cas_num} — image non disponible</div>'

    if WEB_MODE:
        # En mode web, référence relative (images copiées dans le dossier de sortie)
        return (
            f'<img src="images/{cas_num}.png" '
            f'style="max-width:100%; border-radius:8px; margin:8px 0;" '
            f'alt="ECG cas {cas_num}" loading="lazy" />'
        )
    else:
        # Mode standalone : base64 intégré
        import base64
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%; border-radius:8px; margin:8px 0;" '
            f'alt="ECG cas {cas_num}" />'
        )


def _golden_set_html(golden_case: dict) -> str:
    """Mini tableau du golden set pour un cas."""
    rows = []
    for ann in golden_case.get("annotations", []):
        role_raw = ann.get("annotation_role", "")
        if "validant" in role_raw.lower():
            role_badge = '<span style="color:#FFD54F;">🎯 Validant</span>'
        else:
            role_badge = '<span style="color:#CE93D8;">📝 Description</span>'

        rows.append(f"""
        <tr>
            <td style="padding:4px 8px; border-bottom:1px solid #333;">{ann['concept']}</td>
            <td style="padding:4px 8px; border-bottom:1px solid #333; color:#888; font-size:12px;">{ann.get('category', '')}</td>
            <td style="padding:4px 8px; border-bottom:1px solid #333;">{role_badge}</td>
        </tr>
        """)

    return f"""
    <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead>
            <tr style="border-bottom:2px solid #555;">
                <th style="text-align:left; padding:4px 8px; color:#999;">Annotation</th>
                <th style="text-align:left; padding:4px 8px; color:#999;">Catégorie</th>
                <th style="text-align:left; padding:4px 8px; color:#999;">Rôle</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _score_summary_row(cas_num: int, diag: str, report: Optional[CandidateReport]) -> str:
    """Ligne du tableau récapitulatif pour un cas."""
    if report is None:
        return f"""
        <tr>
            <td style="padding:6px 8px; text-align:center; border-bottom:1px solid #333;">{cas_num}</td>
            <td style="padding:6px 8px; border-bottom:1px solid #333;">{diag}</td>
            <td style="padding:6px 8px; text-align:center; border-bottom:1px solid #333; color:#666;">—</td>
            <td style="padding:6px 8px; text-align:center; border-bottom:1px solid #333; color:#666;">Non répondu</td>
        </tr>
        """

    score = report.score_final_pct
    if score >= 90:
        color, emoji = "#4CAF50", "🟢"
    elif score >= 50:
        color, emoji = "#FF9800", "🟠"
    else:
        color, emoji = "#F44336", "🔴"

    return f"""
    <tr>
        <td style="padding:6px 8px; text-align:center; border-bottom:1px solid #333;">{cas_num}</td>
        <td style="padding:6px 8px; border-bottom:1px solid #333;">{diag}</td>
        <td style="padding:6px 8px; text-align:center; border-bottom:1px solid #333;">
            V={report.nb_validants_trouves}/{report.nb_validants_attendus}
        </td>
        <td style="padding:6px 8px; text-align:center; border-bottom:1px solid #333;">
            {emoji} <strong style="color:{color};">{score:.0f}%</strong>
        </td>
    </tr>
    """


def generate_full_html(
    all_results: Dict[str, List[Tuple[int, str, Optional[CandidateReport]]]],
    golden: Dict[int, dict],
) -> str:
    """Génère le document HTML complet avec tous les étudiants."""

    now = datetime.now().strftime("%d/%m/%Y à %H:%M")
    nb_students = len(all_results)

    # ── Calcul des moyennes par étudiant ──────────────────────────────────
    student_averages = {}
    for code, results in all_results.items():
        scores = [r[2].score_final_pct for r in results if r[2] is not None]
        student_averages[code] = sum(scores) / len(scores) if scores else 0

    # ── Début du document ─────────────────────────────────────────────────
    parts = [f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de correction ECG — {nb_students} étudiants × 15 cas</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            background: #121212;
            color: #e0e0e0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #90CAF9;
            font-size: 28px;
            margin-bottom: 4px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-bottom: 32px;
        }}
        h2 {{
            color: #BB86FC;
            border-bottom: 2px solid #333;
            padding-bottom: 8px;
            margin-top: 48px;
            page-break-before: always;
        }}
        h2:first-of-type {{ page-break-before: auto; }}
        h3 {{
            color: #90CAF9;
            margin-top: 24px;
        }}
        .student-header {{
            background: linear-gradient(135deg, #1e3a5f, #1a1a2e);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
            border-left: 4px solid #BB86FC;
        }}
        .student-header h2 {{
            border: none;
            margin: 0;
            padding: 0;
            color: #BB86FC;
            page-break-before: auto;
        }}
        .student-avg {{
            font-size: 36px;
            font-weight: bold;
            margin-top: 8px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 24px;
            font-size: 14px;
        }}
        .summary-table thead th {{
            background: #252525;
            color: #999;
            text-align: left;
            padding: 8px;
            border-bottom: 2px solid #555;
        }}
        .case-block {{
            background: #1e1e1e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            page-break-inside: avoid;
        }}
        .case-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .ecg-container {{
            background: #fff;
            border-radius: 8px;
            padding: 8px;
            margin: 12px 0;
            text-align: center;
        }}
        .ecg-container img {{
            max-width: 100%;
            border-radius: 4px;
        }}
        .student-answer {{
            background: #2d2d2d;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 8px 0;
            font-style: italic;
            color: #bbb;
            border-left: 3px solid #90CAF9;
        }}
        .no-answer {{
            color: #666;
            font-style: italic;
        }}
        .pipeline-result {{
            margin-top: 16px;
        }}
        .nav-toc {{
            background: #1e1e1e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .nav-toc a {{
            color: #90CAF9;
            text-decoration: none;
        }}
        .nav-toc a:hover {{
            text-decoration: underline;
        }}
        .golden-box {{
            background: #252525;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }}
        .golden-label {{
            color: #FFD54F;
            font-weight: bold;
            font-size: 13px;
            margin-bottom: 6px;
        }}
        details {{
            margin: 8px 0;
        }}
        details summary {{
            cursor: pointer;
            padding: 4px;
        }}
        details summary:hover {{
            background: #252525;
            border-radius: 4px;
        }}

        @media print {{
            body {{ background: #121212 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
            .case-block {{ page-break-inside: avoid; }}
            .student-header {{ page-break-before: always; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
<div class="container">

    <h1>🏥 Rapport de Correction ECG</h1>
    <div class="subtitle">
        {nb_students} étudiants × 15 cas — Pipeline RAG Neurosymbolique v1.0<br>
        Généré le {now} — Projet EPCASE
    </div>
"""]

    # ── Table des matières ────────────────────────────────────────────────
    parts.append('<div class="nav-toc no-print">')
    parts.append('<h3 style="color:#FFD54F; margin-top:0;">📑 Table des matières</h3>')
    for code in all_results:
        avg = student_averages.get(code, 0)
        if avg >= 80:
            badge_color = "#4CAF50"
        elif avg >= 50:
            badge_color = "#FF9800"
        else:
            badge_color = "#F44336"
        parts.append(
            f'<div style="padding:4px 0;">'
            f'<a href="#student-{code}">{code}</a> '
            f'— <span style="color:{badge_color}; font-weight:bold;">{avg:.0f}%</span></div>'
        )
    parts.append("</div>")

    # ══════════════════════════════════════════════════════════════════════
    #  Boucle par étudiant
    # ══════════════════════════════════════════════════════════════════════
    for code, results in all_results.items():
        avg = student_averages.get(code, 0)
        if avg >= 80:
            avg_color = "#4CAF50"
        elif avg >= 50:
            avg_color = "#FF9800"
        else:
            avg_color = "#F44336"

        answered = sum(1 for _, _, r in results if r is not None)

        # ── Header étudiant ───────────────────────────────────────────────
        parts.append(f"""
        <div class="student-header" id="student-{code}">
            <h2>👤 Étudiant {code}</h2>
            <div class="student-avg" style="color:{avg_color};">
                Moyenne : {avg:.0f}%
            </div>
            <div style="color:#999; font-size:14px; margin-top:4px;">
                {answered}/15 cas répondus
            </div>
        </div>
        """)

        # ── Tableau récapitulatif ─────────────────────────────────────────
        parts.append("""
        <table class="summary-table">
            <thead>
                <tr>
                    <th style="text-align:center; width:50px;">Cas</th>
                    <th>Diagnostic</th>
                    <th style="text-align:center; width:80px;">Validants</th>
                    <th style="text-align:center; width:100px;">Score</th>
                </tr>
            </thead>
            <tbody>
        """)

        for cas_num, texte, report in results:
            diag = golden.get(cas_num, {}).get("diagnostic_principal", "?")
            parts.append(_score_summary_row(cas_num, diag, report))

        parts.append("</tbody></table>")

        # ── Détail cas par cas ────────────────────────────────────────────
        for cas_num, texte, report in results:
            cas_info = golden.get(cas_num, {})
            diag = cas_info.get("diagnostic_principal", "?")
            cat = cas_info.get("category", "")

            parts.append(f'<div class="case-block">')

            # Header du cas
            if report:
                score = report.score_final_pct
                if score >= 90:
                    sc, se = "#4CAF50", "🟢"
                elif score >= 50:
                    sc, se = "#FF9800", "🟠"
                else:
                    sc, se = "#F44336", "🔴"
                score_html = f'{se} <strong style="color:{sc}; font-size:24px;">{score:.0f}%</strong>'
            else:
                score_html = '<span style="color:#666;">—</span>'

            parts.append(f"""
            <div class="case-header">
                <h3 style="margin:0;">Cas {cas_num} — {diag}</h3>
                {score_html}
            </div>
            <div style="color:#888; font-size:12px; margin-bottom:8px;">Catégorie : {cat}</div>
            """)

            # ECG image
            parts.append(f"""
            <details>
                <summary style="color:#90CAF9; font-size:13px;">📸 Voir l'ECG</summary>
                <div class="ecg-container">
                    {_ecg_image_tag(cas_num)}
                </div>
            </details>
            """)

            # Golden set (repliable)
            parts.append(f"""
            <details>
                <summary style="color:#FFD54F; font-size:13px;">🎯 Correction attendue (golden set)</summary>
                <div class="golden-box">
                    {_golden_set_html(cas_info)}
                </div>
            </details>
            """)

            # Réponse de l'étudiant
            if texte:
                parts.append(f"""
                <div style="margin-top:12px;">
                    <div style="color:#90CAF9; font-weight:bold; font-size:13px; margin-bottom:4px;">
                        📝 Réponse de l'étudiant
                    </div>
                    <div class="student-answer">« {texte} »</div>
                </div>
                """)
            else:
                parts.append('<div class="no-answer">⬜ Pas de réponse pour ce cas</div>')

            # Résultat pipeline (rapport complet HTML)
            if report:
                parts.append('<div class="pipeline-result">')
                parts.append(format_report_html(report))
                parts.append("</div>")

            parts.append("</div>")  # fin case-block

    # ── Footer ────────────────────────────────────────────────────────────
    parts.append(f"""
    <div style="text-align:center; color:#555; font-size:12px; margin-top:48px; padding:24px;">
        <hr style="border-color:#333;">
        Rapport généré automatiquement par le Pipeline RAG Neurosymbolique v1.0<br>
        Projet EPCASE — edu-ecg — {now}
    </div>

</div>
</body>
</html>
""")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Génère un rapport HTML de correction ECG")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Chemin vers le CSV des réponses étudiants (par défaut : lit les JSON)",
    )
    parser.add_argument(
        "--students", nargs="*", default=None,
        help="Codes étudiants à inclure (ex: ECG-WY55 ECG-7512). Par défaut : tous.",
    )
    parser.add_argument(
        "--feedback", action="store_true", default=False,
        help="Activer le feedback pédagogique GPT (plus lent)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Chemin du fichier HTML de sortie",
    )
    parser.add_argument(
        "--web", action="store_true", default=False,
        help="Mode web : images séparées (site léger ~5 Mo, prêt pour GitHub Pages / Scalingo)",
    )
    args = parser.parse_args()

    # Activer le mode web si demandé
    global WEB_MODE
    WEB_MODE = args.web

    print("=" * 60)
    print("📄 Génération du rapport de correction ECG")
    print("=" * 60)

    # 1. Charger le golden set
    print("\n📥 Chargement du golden set...")
    golden = load_golden_set()
    print(f"   ✅ {len(golden)} cas chargés")

    # 2. Charger les étudiants (JSON par défaut, CSV si --csv)
    if args.csv:
        csv_path = Path(args.csv)
        print(f"\n📥 Chargement du CSV : {csv_path.name}")
        students = load_student_csv(csv_path)
    else:
        print(f"\n📥 Chargement des JSON étudiants : {STUDENTS_DIR}")
        students = load_student_json()
    print(f"   ✅ {len(students)} étudiants trouvés")

    # Filtrer si demandé
    if args.students:
        students = {k: v for k, v in students.items() if k in args.students}
        print(f"   📌 Filtré → {len(students)} étudiants : {', '.join(students.keys())}")

    if not students:
        print("   ❌ Aucun étudiant à traiter !")
        return

    # 3. Évaluer chaque étudiant
    print(f"\n🔄 Évaluation de {len(students)} étudiants × 15 cas...")
    print(f"   Feedback pédagogique : {'✅ Activé' if args.feedback else '❌ Désactivé (mode rapide)'}")
    print()

    all_results: Dict[str, List[Tuple[int, str, Optional[CandidateReport]]]] = {}
    t_global = time.time()

    for i, (code, responses) in enumerate(students.items(), 1):
        answered = len(responses)
        print(f"   [{i}/{len(students)}] 👤 {code} ({answered} réponses)...", end="", flush=True)
        t0 = time.time()

        results = evaluate_student(code, responses, golden, with_feedback=args.feedback)
        all_results[code] = results

        # Résumé rapide
        scores = [r[2].score_final_pct for r in results if r[2] is not None]
        avg = sum(scores) / len(scores) if scores else 0
        dt = time.time() - t0

        print(f" ✅ {avg:.0f}% moyen ({dt:.1f}s)")

    total_time = time.time() - t_global
    print(f"\n   ⏱️ Total : {total_time:.1f}s")

    # 4. Générer les rapports individuels dans rapports_v2/
    rapports_v2_dir = EVAL_ROOT / "rapports_v2"
    rapports_v2_dir.mkdir(exist_ok=True)
    print(f"\n📝 Génération des rapports individuels dans {rapports_v2_dir}...")
    for code, results in all_results.items():
        individual_html = generate_full_html({code: results}, golden)
        indiv_path = rapports_v2_dir / f"rapport_{code}.html"
        with open(indiv_path, "w", encoding="utf-8") as f:
            f.write(individual_html)
        print(f"   ✅ {indiv_path.name}")
    print(f"   📁 {len(all_results)} rapports individuels écrits")

    # 5. Générer le HTML groupé
    print("\n📝 Génération du rapport groupé...")
    html = generate_full_html(all_results, golden)

    # 6. Écrire le fichier groupé
    EXPORTS_DIR.mkdir(exist_ok=True)

    if args.web:
        # Mode web : créer un dossier site/ avec index.html + images/
        site_dir = EXPORTS_DIR / "site"
        site_dir.mkdir(exist_ok=True)
        images_dir = site_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Copier les 15 images ECG (une seule fois)
        print("   📸 Copie des images ECG...")
        for i in range(1, 16):
            src = IMAGES_ROOT / f"{i}.png"
            dst = images_dir / f"{i}.png"
            if src.exists():
                shutil.copy2(src, dst)
        
        output_path = site_dir / "index.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        size_mb = sum(fp.stat().st_size for fp in site_dir.rglob("*") if fp.is_file()) / (1024 * 1024)
        print(f"\n✅ Site web écrit : {site_dir}")
        print(f"   📏 Taille totale : {size_mb:.1f} Mo")
        print(f"   📄 {output_path}")
        print(f"   📸 {images_dir} ({len(list(images_dir.glob('*.png')))} images)")
        print(f"\n   🚀 Prêt pour déploiement GitHub Pages / Scalingo / Firebase")
        print(f"   💡 Tester localement : ouvrir {output_path} dans un navigateur")

    else:
        if args.output:
            output_path = Path(args.output)
        else:
            date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
            output_path = EXPORTS_DIR / f"rapport_corrections_{date_str}.html"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Rapport écrit : {output_path}")
        print(f"   📏 Taille : {size_mb:.1f} Mo")
        print(f"   🌐 Ouvrez dans un navigateur et faites Ctrl+P pour exporter en PDF")

    print()
    return str(output_path)


if __name__ == "__main__":
    main()
