# 🧪 Données de test

Données réelles (anonymisées) pour tester le pipeline end-to-end.

## 📁 Structure

```
testing_data/
├── goldenset/                      ← 15 cas annotés par cardiologue expert
│   └── case_<id>/
│       ├── ecg_1.png               ← Image ECG source (~2 Mo)
│       └── metadata.json           ← Annotations : expected_concepts,
│                                     diagnostic_principal, exclusions, etc.
│
└── corrections/                    ← 43 corrections étudiantes déjà évaluées
    ├── ECG-XXXX.json               ← Texte de l'étudiant + sortie complète
    │                                 du pipeline V3 (concepts, scores, traces)
    ├── golden.json                 ← Golden set agrégé (format Streamlit)
    └── data.json                   ← Config session
```

## 🚀 Tester le pipeline sur un cas du goldenset

```bash
python examples/03_run_on_goldenset.py
```

Ce script :
1. Charge un cas du goldenset (image + concepts attendus)
2. Prend le texte d'un étudiant qui a répondu à ce cas (depuis `corrections/`)
3. Relance le pipeline V3 dessus
4. Affiche le rapport + compare au score original

## 📊 Structure d'un cas du goldenset

```json
{
  "case_id": "case_20260223_220733_b885d5ef",
  "name": "2",
  "category": "Troubles du Rythme",
  "difficulty": "🟢 Débutant",
  "annotations": [
    {
      "concept": "BAV complet",
      "category": "DIAGNOSTIC_URGENT",
      "coefficient": 1.0,
      "annotation_role": "🎯 Diagnostic validant"
    }
  ],
  "expected_concepts": ["BAV complet"],
  "diagnostic_principal": "BAV complet",
  "ecgs": [{"filename": "ecg_1.png", "index": 1}]
}
```

## 📊 Structure d'une correction étudiante

```json
{
  "code": "ECG-0NAW",
  "pipeline_version": "RAG Neurosymbolique v1.0",
  "average": 62.4,
  "cases": {
    "2": {
      "student_text": "Fréquence à 50 bpm, rythme sinusal...",
      "report": {
        "score_final_pct": 0.0,
        "concepts_extraits": [...],     ← Chaque entité extraite avec
                                          method, ontology_id, top_k_candidats
        "validant_details": [...],      ← Concepts attendus trouvés/manqués
        "decouvertes": [...]            ← Concepts vrais découverts en plus
      }
    }
  }
}
```

## 🔒 Anonymisation

Toutes les données ont été anonymisées :
- Codes étudiants pseudonymisés (`ECG-XXXX`, 4 caractères aléatoires)
- Images ECG dépourvues d'identifiants patient
- Aucune information nominative dans les textes
