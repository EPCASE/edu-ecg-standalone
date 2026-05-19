# 🩺 Edu-ECG — Pipeline RAG Neurosymbolique V3

Pipeline d'évaluation automatique des comptes rendus ECG d'étudiants en médecine,
basé sur une **ontologie OWL** (345 concepts) et un **RAG hybride** (Dense + BM25)
contrôlé par un **juge LLM** (GPT-4o-mini).

> **Standalone fonctionnel** — Cloner, installer, configurer la clé OpenAI, et lancer.
> Pas de dépendance à un autre dépôt. L'index RAG (3.9 Mo) est inclus pré-calculé.

---

## ⚡ Démarrage rapide (5 minutes)

```bash
# 1. Cloner le dépôt
git clone https://github.com/<owner>/edu-ecg-standalone.git
cd edu-ecg-standalone

# 2. Créer un environnement Python
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé OpenAI
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
# Puis éditer .env et y mettre votre OPENAI_API_KEY

# 5. Tester le scoring sans appel LLM (gratuit, instantané)
python examples/02_score_only.py

# 6. Tester le pipeline complet (avec appels GPT, ~0.02$/run)
python examples/01_run_pipeline_v3.py
```

---

## 📁 Structure du dépôt

```
edu-ecg-standalone/
├── rag_pipeline/                ← Le pipeline V3 complet (6 briques)
│   ├── ontology_index.py        ← Brique 1 : construction index vectoriel
│   ├── ner_extractor.py         ← Brique 2 : NER GPT-4o (Structured Outputs)
│   ├── hybrid_search.py         ← Brique 3 : Dense + BM25 + RRF
│   ├── neurosymbolic_judge.py   ← Brique 4 : Juge LLM contraint
│   ├── scoring_v3.py            ← Brique 5 : Scoring hiérarchique ontologique
│   ├── pedagogical_feedback.py  ← Brique 6 : Feedback pédagogique GPT
│   ├── semantic_layer.py        ← Accès uniforme à l'ontologie V2
│   ├── candidate_report.py      ← Orchestrateur (entrée principale)
│   ├── generate_html_report.py  ← Export HTML d'un rapport étudiant
│   ├── edn_knowledge_base.py    ← Base de connaissances EDN (fiches concepts)
│   ├── data/
│   │   ├── ontology_v2.json     ← Ontologie 345 concepts (poids, hiérarchie, exclusions)
│   │   ├── ontology_from_owl.json
│   │   └── ontology_source.owl  ← Source WebProtégé
│   └── rag_index/               ← Index pré-calculé (3.9 Mo)
│       ├── vecteurs_ontologie.npy        (658 vecteurs × 1536 dims)
│       ├── metadata_ontologie.json
│       └── bm25_corpus.json
│
├── scripts/                     ← Scripts utilitaires (régénération)
│   ├── regenerate_ontology.py   ← OWL → ontology_from_owl.json
│   ├── convert_owl_to_v2.py     ← ontology_from_owl → ontology_v2 (hiérarchie + poids)
│   └── rdf_owl_extractor.py     ← Backend OWL/RDF
│
├── examples/                    ← Exemples prêts à l'emploi
│   ├── 01_run_pipeline_v3.py    ← Pipeline complet sur un texte
│   └── 02_score_only.py         ← Scoring V3 seul (sans LLM, sans clé)
│
├── requirements.txt
├── .env.example
├── .gitignore
├── ontology_source.owl          ← 🦉 Ontologie OWL source (WebProtégé, 431 Ko)
│                                   À explorer avec Protégé ou éditer en ligne
├── ARCHITECTURE.md              ← Schéma général de la plateforme
├── ARCHITECTURE_PIPELINE.md     ← Détail des 6 briques
└── README.md                    ← Ce fichier
```

---

## 🧠 Architecture du pipeline

```
Texte étudiant
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  Brique 2 — NER (GPT-4o, Structured Outputs)                  │
│  → Extrait les entités cliniques + statut (present/absent/hyp)│
└──────────────────────┬────────────────────────────────────────┘
                       │ entités brutes
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  Brique 3 — Recherche hybride                                 │
│  Dense (embeddings 1536d) + BM25 + Reciprocal Rank Fusion     │
│  → Top-K candidats de l'ontologie                             │
└──────────────────────┬────────────────────────────────────────┘
                       │ candidats
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  Brique 4 — Juge neurosymbolique                              │
│  Coupe-circuit (matching exact, 42% des cas, gratuit)         │
│  + Juge LLM contraint (GPT-4o-mini) sur les candidats         │
│  → ID ontologique validé (ou NONE)                            │
└──────────────────────┬────────────────────────────────────────┘
                       │ IDs résolus
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  Brique 5 — Scoring V3 ontologique (scoring_v3.py)            │
│  Match exact / requires / qualifier / support / excluded /    │
│  hiérarchique (parent/enfant) + conversion des négations      │
│  → Score 0-100% par concept attendu                           │
└──────────────────────┬────────────────────────────────────────┘
                       │ scores
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  Brique 6 — Feedback pédagogique (GPT-4o-mini)                │
│  → Commentaire personnalisé pour l'étudiant                   │
└───────────────────────────────────────────────────────────────┘
```

---

## 🔧 Utilisation programmatique

### Pipeline complet (avec LLM)

```python
from candidate_report import generate_candidate_report, format_report_text

report = generate_candidate_report(
    texte_etudiant="QRS large à 140 ms, aspect en M en V5-V6, BBG complet.",
    golden_names=["Bloc de branche gauche complet"],
    golden_ids=["BLOC_DE_BRANCHE_GAUCHE_COMPLET"],
    golden_roles=["validant"],
    with_feedback=True,
)

print(format_report_text(report))
print(f"Score : {report.score_final_pct}%")
```

### Scoring V3 seul (déterministe, sans LLM)

```python
from scoring_v3 import score_student_response_v3, format_v3_summary

result = score_student_response_v3(
    found_ids=["BLOC_DE_BRANCHE_GAUCHE_COMPLET"],
    expected_ids=["BLOC_DE_BRANCHE_GAUCHE_COMPLET"],
)
print(format_v3_summary(result))
print(f"Score : {result.score_pct}%")
```

---

## � Explorer / éditer l'ontologie

Le fichier **`ontology_source.owl`** à la racine du dépôt est l'ontologie OWL source
(345 concepts, ~431 Ko, exportée depuis WebProtégé). Elle peut être :

- **Visualisée** dans [Protégé Desktop](https://protege.stanford.edu/) (gratuit, multiplateforme)
- **Éditée collaborativement** sur [WebProtégé](https://webprotege.stanford.edu/)
- **Re-générée** en JSON via `python scripts/regenerate_ontology.py` puis `python scripts/convert_owl_to_v2.py`

```
OWL (WebProtégé)
   │
   ▼  scripts/regenerate_ontology.py  (rdflib)
ontology_from_owl.json   (plat : ID + nom + synonymes)
   │
   ▼  scripts/convert_owl_to_v2.py    (enrichissement)
ontology_v2.json         (hiérarchie + poids + exclusions + requires/qualifier/support)
   │
   ▼  rag_pipeline/ontology_index.py  (embeddings OpenAI)
rag_index/               (vecteurs + BM25 + métadonnées)
```

---

## �🧪 Régénérer l'index RAG (si l'ontologie change)

```bash
cd rag_pipeline
python ontology_index.py
```

Cela reconstruit `rag_index/` à partir de `data/ontology_v2.json`.
Coût : ~$0.01 d'embeddings OpenAI, ~30 secondes.

---

## 💰 Coûts OpenAI

| Brique | Modèle | Coût approximatif |
|---|---|---|
| 2 — NER | `gpt-4o-2024-08-06` | ~$0.005 / texte étudiant |
| 4 — Juge | `gpt-4o-mini` | ~$0.001 / terme non résolu par coupe-circuit |
| 6 — Feedback | `gpt-4o-mini` | ~$0.005 / rapport |
| Index | `text-embedding-3-small` | ~$0.01 (à la construction uniquement) |

**Total : ~$0.02 par évaluation complète d'un texte étudiant.**

---

## 📚 Documentation complémentaire

- `ARCHITECTURE.md` — Vue d'ensemble de la plateforme Edu-ECG
- `ARCHITECTURE_PIPELINE.md` — Détails techniques des 6 briques
- `rag_pipeline/README_PIPELINE.md` — Doc historique du pipeline

---

## 🔒 Données privées (non incluses dans ce dépôt)

Ce dépôt ne contient **que le code et l'ontologie**. Les données sensibles
(images ECG, cas du golden set, corrections étudiantes) sont fournies séparément
sous forme de zip privé. Voir `.gitignore` pour la liste des dossiers exclus.

---

## 📝 Licence

MIT — Voir `LICENSE`

## 👥 Auteurs

- Grégoire Massoullié — CHU Clermont-Ferrand
- Équipe BMad
