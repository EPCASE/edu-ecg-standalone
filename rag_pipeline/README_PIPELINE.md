# 🧠 Pipeline RAG Neurosymbolique — Évaluation ECG

> **Branche** : `RAGontologique`  
> **Repo** : [EPCASE/edu-ecg](https://github.com/EPCASE/edu-ecg)  
> **Date** : Février 2026

## Vue d'ensemble

Pipeline hybride **ontologie + LLMs** pour évaluer automatiquement les interprétations ECG d'étudiants en médecine. L'architecture repose sur 5 briques séquentielles qui combinent raisonnement symbolique (ontologie OWL) et inférence neuronale (GPT-4o / GPT-4o-mini).

```
Texte étudiant ──→ [Brique 2: NER] ──→ [Brique 3: Search] ──→ [Brique 4: Juge] ──→ Score
                    GPT-4o              Dense+BM25+RRF          Coupe-circuit +
                    Structured          sur index vectoriel     GPT-4o-mini QCM
                    Outputs             ontologique
```

### Résultats Benchmark

| Métrique | v1 | v2 | Delta |
|---|---|---|---|
| **Score moyen** | 48.9% | **62.4%** | +13.5 pts |
| **Score médian** | 50.0% | **86.2%** | +36.2 pts |
| Cas parfaits (100%) | — | 29/73 | |
| Cas à 0% | — | 18/73 | |

*Benchmark sur 5 cardiologues × 15 cas ECG (73 évaluations exploitables).*

---

## Architecture en 5 Briques

### 🧱 Brique 1 — Socle Symbolique & Vectoriel (`ontology_index.py`)

**Rôle** : Transformer l'ontologie OWL en base vectorielle locale pour recherche instantanée en RAM.

**Entrée** : `ontology_from_owl.json` (289 concepts, 202 synonymes, extraits de l'OWL via `rdf_owl_extractor.py`)

**Sortie** :
- `vecteurs_ontologie.npy` — matrice 491×1536 (float32, text-embedding-3-small)
- `metadata_ontologie.json` — registre index ↔ document avec ontology_id, surface_form, catégorie, poids
- `bm25_corpus.json` — corpus tokenisé pour BM25

**Chaque concept** génère N documents indexés :
- Le nom canonique (`"Fibrillation atriale"`)
- Chaque synonyme (`"FA"`, `"ACFA"`, `"Fibrillation auriculaire"`)

**Normalisation** : accents supprimés, lowercase, espaces normalisés → permet le matching exact.

---

### 🧱 Brique 2 — Extracteur NER (`ner_extractor.py`)

**Rôle** : Extraire toutes les entités cliniques du texte libre d'un étudiant.

**Modèle** : `gpt-4o-2024-08-06` avec **Structured Outputs** (schéma Pydantic garanti)

**Stratégie d'interaction LLM** :

Le prompt système impose une **extraction pure sans normalisation** :
- Les termes sont extraits tels quels (fautes d'orthographe incluses : `"tachi supra"` → `"tachi supra"`)
- Le périmètre couvre l'ECG au sens large : morphologie, rythme, diagnostics, **diagnostics étiologiques** (hyperkaliémie, embolie pulmonaire, amylose…), stimulation cardiaque
- Chaque entité porte un **statut clinique** : `present` / `absent` / `hypothese`

```python
class ClinicalEntity(BaseModel):
    terme_brut: str          # "tachi supra" (tel quel)
    statut: Literal["present", "absent", "hypothese"]
    contexte_phrase: str     # phrase d'origine complète
```

**Pourquoi ce design** : L'extraction brute délègue toute la normalisation au Search (Brique 3) et la décision au Juge (Brique 4). Cela évite le double jeu de devinettes (GPT-4o qui essaierait de mapper directement vers l'ontologie).

---

### 🧱 Brique 3 — Recherche Hybride (`hybrid_search.py`)

**Rôle** : Pour chaque terme NER brut, trouver les Top-K concepts ontologiques les plus proches.

**Deux moteurs combinés** :
1. **Dense (sémantique)** : embedding du terme via `text-embedding-3-small` → cosinus avec la matrice de Brique 1
2. **Sparse (lexical)** : BM25Okapi sur les surface_forms normalisées → excellent pour les acronymes (`"BBG"`, `"FA"`, `"TV"`)

**Fusion** : **Reciprocal Rank Fusion (RRF)** avec boost BM25 pour les acronymes courts (≤5 chars).

**Flag `is_exact_match`** : si la forme normalisée du terme query correspond exactement à une surface_form indexée → flag booléen transmis à la Brique 4 pour le coupe-circuit.

```python
moteur.search_top_k("FA", k=5)
# → [{"ontology_id": "FIBRILLATION_ATRIALE", "is_exact_match": True, ...}, ...]
```

---

### 🧱 Brique 4 — Le Juge Neurosymbolique (`neurosymbolic_judge.py`)

**Rôle** : Décision finale — relier un terme brut à un `ontology_id` ou `"NONE"`.

**Pipeline en 2 étapes** :

#### Étape 1 : Coupe-Circuit (bypass LLM)
Si le candidat n°1 a `is_exact_match=True` → retour immédiat de son `ontology_id`.

**⚡ ~42% des résolutions passent par le coupe-circuit** (0 token LLM, latence ~0ms).

**Garde-fou spécificité** : si le match exact est un descripteur générique (poids=1) ET qu'un concept plus spécifique (poids>1, même préfixe d'ID) existe dans le Top-K → le coupe-circuit est **annulé** et le Juge LLM est invoqué.

> Exemple : `"Tachycardie"` match exact `TACHYCARDIE` (p=1), mais `TACHYCARDIE_VENTRICULAIRE` (p=4) est candidat #2 → le Juge décide avec le contexte.

#### Étape 2 : Juge LLM (QCM)
Le Top-K est présenté à `gpt-4o-mini` sous forme de **QCM** (Question à Choix Multiple). Le LLM doit choisir un ID parmi les candidats ou répondre `"NONE"`.

**Stratégie d'interaction LLM** :

Le prompt système encode 4 règles strictes par ordre de priorité :

1. **Contexte** : analyser le terme ET sa phrase d'origine
2. **Options only** : choisir UNIQUEMENT parmi les candidats fournis
3. **Spécificité maximale** : toujours préférer le concept le plus spécifique (enfant > parent)
4. **Diagnostic > Descripteur** : préférer les diagnostics (poids≥3) sur les descripteurs/territoires

```
Candidats de l'ontologie proposés :
- Flutter droit typique (ID: FLUTTER_DROIT_TYPIQUE) [catégorie: DIAGNOSTIC_MAJEUR, poids: 3]
- Flutter atrial (ID: FLUTTER_ATRIAL) [catégorie: SIGNE_ECG_PATHOLOGIQUE, poids: 2]
- Flutter atrial atypique (ID: FLUTTER_ATRIAL_ATYPIQUE) [catégorie: DIAGNOSTIC_MAJEUR, poids: 3]
```

**Réponse structurée** (Pydantic) :
```python
class ConceptMatching(BaseModel):
    id_ontologie: str     # "FLUTTER_DROIT_TYPIQUE" ou "NONE"
    justification: str    # "Flutter typique → concept spécifique..."
```

**Validation** : si le LLM renvoie un ID absent des candidats → forçage `NONE`.

---

### 🧱 Brique 5 — Scoring (`correction_llm.py`)

**Rôle** : Comparer les IDs ontologiques trouvés vs le Golden Set expert.

**Scoring pondéré** :
- Chaque concept du golden set a un **poids** (1 = descripteur, 2 = signe, 3 = diagnostic majeur, 4 = diagnostic urgent)
- Score = Σ(poids_validés) / Σ(poids_attendus) × 100
- **Bonus diagnostic** : +15% si au moins un concept de poids ≥ 3 est trouvé
- **Statut hypothèse** : pondéré à 80% au lieu de 100%
- **Implications automatiques** : si un concept parent est trouvé, certains enfants sont automatiquement validés (ex: `BAV_COMPLET` implique `BAV`)

---

## Stratégie d'Interaction Ontologie ↔ LLMs

### Principe fondamental : **séparation des responsabilités**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ONTOLOGIE (symbolique)                           │
│  • Source de vérité : 289 concepts, poids, catégories, synonymes   │
│  • Index vectoriel : 491 documents (embeddings 1536-dim)           │
│  • Règles d'implication : inférence déterministe                   │
│  • Normalisation de texte : matching exact sans ambiguïté           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                     Candidats Top-K
                     + métadonnées
                     (poids, catégorie,
                      is_exact_match)
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      LLMs (neuronal)                                │
│  • Brique 2 (GPT-4o) : extraction NER brute — AUCUNE ontologie     │
│  • Brique 4 (GPT-4o-mini) : QCM contraint — ontologie comme garde  │
│                                                                      │
│  Le LLM ne voit JAMAIS l'ontologie complète.                        │
│  Il voit uniquement les Top-K candidats pré-filtrés.                │
│  Il ne peut PAS inventer un ID — validation post-LLM stricte.       │
└─────────────────────────────────────────────────────────────────────┘
```

### Les 3 interactions clés

| Interaction | Qui → Qui | Mécanisme | Objectif |
|---|---|---|---|
| **NER → Search** | LLM → Ontologie | Terme brut → embedding → cosinus + BM25 | Trouver des candidats (tolérance aux fautes) |
| **Search → Juge** | Ontologie → LLM | Top-K candidats (avec poids/catégorie) → QCM | Décision clinique contextuelle |
| **Coupe-circuit** | Ontologie seule | Matching exact normalisé → bypass LLM | Rapidité + déterminisme (42% des cas) |

### Pourquoi cette architecture ?

1. **Le LLM n'a pas accès à l'ontologie complète** → il ne peut pas halluciner un concept inexistant
2. **Le coupe-circuit est déterministe** → même entrée = même sortie, pas de variabilité LLM
3. **Le scoring est symbolique** → poids/catégories gérés par l'ontologie OWL, pas par le LLM
4. **Le NER est découplé** → l'extraction brute est indépendante de l'ontologie, ce qui rend le pipeline robuste aux mises à jour de l'OWL
5. **Le Juge voit les métadonnées** → poids et catégorie guident le LLM vers le bon choix sans hardcoder de règles

### Garde-fous contre les erreurs LLM

| Garde-fou | Couche | Mécanisme |
|---|---|---|
| **Structured Outputs** | Brique 2 & 4 | Schéma Pydantic garanti par l'API OpenAI |
| **Validation post-LLM** | Brique 4 | L'ID renvoyé doit être dans les candidats soumis |
| **Coupe-circuit garde-fou** | Brique 4 | Annule le bypass si un concept plus spécifique existe |
| **Implication rules** | Brique 5 | Validation automatique de concepts implicites |
| **Forçage NONE** | Brique 4 | Si ID invalide → NONE (plutôt qu'un faux positif) |

---

## Fichiers du pipeline

```
RAG ontologique/
├── ontology_index.py          # Brique 1 — Construction index vectoriel
├── ner_extractor.py           # Brique 2 — NER GPT-4o
├── hybrid_search.py           # Brique 3 — Recherche hybride Dense+BM25+RRF
├── neurosymbolic_judge.py     # Brique 4 — Juge neurosymbolique
├── rag_index/                 # Index pré-calculé (491 documents × 1536 dims)
│   ├── vecteurs_ontologie.npy
│   ├── metadata_ontologie.json
│   └── bm25_corpus.json
├── tests/
│   └── benchmark_evaluation.ipynb  # Benchmark complet (5 participants × 15 cas)
├── test_brique1.py            # Tests unitaires
├── test_brique2.py
├── test_brique3.py
├── test_brique4.py
└── SUGGESTIONS_ONTOLOGIE_2026-02-26.md  # Suggestions d'enrichissement

ECG lecture/
├── BrYOzRZIu7jQTwmfcGsi35.owl         # Ontologie OWL source (WebProtégé)
├── data/ontology_from_owl.json          # Ontologie JSON (289 concepts, 202 synonymes)
├── regenerate_ontology.py               # Script de régénération JSON depuis OWL
├── backend/rdf_owl_extractor.py         # Extracteur OWL → JSON
└── frontend/pages/correction_llm.py     # Scoring de production (Brique 5)
```

---

## Modèles utilisés

| Modèle | Usage | Coût approx. |
|---|---|---|
| `text-embedding-3-small` | Embeddings index + requêtes (1536 dims) | ~$0.02/1M tokens |
| `gpt-4o-2024-08-06` | NER Structured Outputs (Brique 2) | ~$2.50/1M input |
| `gpt-4o-mini` | Juge QCM (Brique 4) | ~$0.15/1M input |

---

## Exécution

```bash
# 1. Régénérer l'ontologie JSON depuis l'OWL
cd "ECG lecture"
python regenerate_ontology.py

# 2. Reconstruire l'index vectoriel
cd "RAG ontologique"
python ontology_index.py

# 3. Lancer le benchmark
# → Ouvrir tests/benchmark_evaluation.ipynb et exécuter les cellules 1→4
```

## Prérequis

- Python 3.12+
- `OPENAI_API_KEY` dans un fichier `.env`
- Packages : `openai`, `numpy`, `pydantic`, `python-dotenv`, `pandas`, `tqdm`
