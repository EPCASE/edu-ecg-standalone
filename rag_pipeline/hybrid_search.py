"""
🧱 Brique 3 — Recherche Hybride (Dense + Sparse + RRF)
========================================================
Prend un terme brut extrait par GPT-4o (Brique 2) et trouve les Top-K
meilleurs concepts correspondants dans l'ontologie locale.

Deux moteurs combinés :
  - Dense  (sémantique) : embedding OpenAI text-embedding-3-small + cosinus
  - Sparse (lexical)     : BM25Okapi sur les surface_forms normalisés

Fusion via Reciprocal Rank Fusion (RRF) avec boost acronyme BM25.

Dépendances :
  - vecteurs_ontologie.npy   (matrice N×1536, produite par Brique 1)
  - metadata_ontologie.json  (registre i ↔ doc i, produit par Brique 1)
  - normalize_text / tokenize (fonctions de Brique 1)
  - OpenAI API (text-embedding-3-small) pour vectoriser la requête

Auteur : BMad Team
Date   : 2026-02-25
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

# Import des utilitaires de normalisation de la Brique 1
from ontology_index import normalize_text, tokenize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flexion helper — normalise les variantes pluriel/genre du français médical
# ---------------------------------------------------------------------------

def _deflect(text: str) -> set:
    """
    Génère les variantes flexionnelles d'un texte normalisé pour
    absorber les différences singulier/pluriel et masculin/féminin
    dans le test d'exact match.

    Ex: "qrs larges" → {"qrs large"}
        "flutter atriale" → {"flutter atrial"}
        "ondes t amples" → {"onde t ample", "ondes t ample", ...}

    Ne touche PAS à normalize_text ni au BM25/embeddings,
    s'applique uniquement au test is_exact_match.
    """
    variants = set()
    words = text.split()
    if not words:
        return variants

    # Pour chaque mot, générer la variante sans -s, -x, -es, -e final
    for i, w in enumerate(words):
        new_words = list(words)
        # Pluriel : -s, -x
        if w.endswith("s") and len(w) > 2 and not w.endswith("ss"):
            new_words[i] = w[:-1]
            variants.add(" ".join(new_words))
        if w.endswith("x") and len(w) > 2:
            new_words[i] = w[:-1]
            variants.add(" ".join(new_words))
        # Pluriel -es → -e (ex: "larges" → "large")
        # Déjà couvert par le -s ci-dessus
        # Genre : -e final (ex: "atriale" → "atrial")
        if w.endswith("e") and len(w) > 3 and not w.endswith("ee"):
            new_words[i] = w[:-1]
            variants.add(" ".join(new_words))
        # Genre inverse : ajouter -e (ex: "atrial" → "atriale")
        if not w.endswith("e") and len(w) > 3:
            new_words[i] = w + "e"
            variants.add(" ".join(new_words))

    return variants


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
# Classe principale — Brique 3
# ---------------------------------------------------------------------------

class HybridSearchEngine:
    """
    Moteur de recherche hybride (Dense + Sparse) sur l'ontologie ECG.

    Initialisé une seule fois au démarrage, il charge en RAM :
      - la matrice d'embeddings (411×1536, ~2.5 Mo)
      - les métadonnées des documents
      - l'index BM25

    La méthode .search_top_k() est ensuite ultra-rapide (~50 ms par requête,
    dominé par l'appel API OpenAI pour l'embedding de la query).

    Usage:
        engine = HybridSearchEngine("rag_index/")
        results = engine.search_top_k("tachi supra", k=5)
        for r in results:
            print(r["ontology_id"], r["surface_form"], r["score"])
    """

    # Constantes RRF
    RRF_K = 60          # constante standard de la RRF
    BM25_BOOST = 1.5    # boost BM25 pour les matchs exacts (acronymes)

    # Modèle d'embedding (doit correspondre à celui de Brique 1)
    EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(self, index_dir: str = "rag_index/"):
        """
        Charge l'index pré-calculé depuis le disque.

        Args:
            index_dir: Répertoire contenant vecteurs_ontologie.npy
                       et metadata_ontologie.json.
        """
        index_path = Path(index_dir)

        # --- 1. Chargement des embeddings ---
        npy_path = index_path / "vecteurs_ontologie.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Matrice d'embeddings introuvable : {npy_path}")
        self.embeddings: np.ndarray = np.load(npy_path)

        # --- 2. Chargement des métadonnées ---
        json_path = index_path / "metadata_ontologie.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Métadonnées introuvables : {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.documents: List[Dict] = meta["documents"]
        self.index_meta: Dict = {k: v for k, v in meta.items() if k != "documents"}

        assert len(self.documents) == self.embeddings.shape[0], (
            f"Incohérence : {len(self.documents)} documents vs "
            f"{self.embeddings.shape[0]} lignes d'embeddings"
        )

        # --- 3. Construction de l'index BM25 ---
        # Tokeniser chaque surface_form avec la même normalisation que Brique 1
        self._bm25_corpus: List[List[str]] = [
            tokenize(doc["surface_form"]) for doc in self.documents
        ]
        self._bm25 = BM25Okapi(self._bm25_corpus)

        # --- 4. Index inverse : ontology_id → set(surface_forms normalisées) ---
        # Utilisé par la Brique 4 (coupe-circuit) pour vérifier les matchs exacts
        # contre TOUTES les formes d'un concept (canonical + synonymes)
        self._normalized_forms_by_id: Dict[str, set] = {}
        for doc in self.documents:
            oid = doc["ontology_id"]
            norm = normalize_text(doc["surface_form"])
            if oid not in self._normalized_forms_by_id:
                self._normalized_forms_by_id[oid] = set()
            self._normalized_forms_by_id[oid].add(norm)

        logger.info(
            f"🔍 HybridSearchEngine initialisé : "
            f"{len(self.documents)} documents, "
            f"embeddings {self.embeddings.shape}, "
            f"modèle {self.index_meta.get('embedding_model', self.EMBEDDING_MODEL)}"
        )

    # ------------------------------------------------------------------
    # Accès aux formes normalisées d'un concept (pour Brique 4)
    # ------------------------------------------------------------------

    def get_all_normalized_forms(self, ontology_id: str) -> set:
        """
        Retourne l'ensemble des surface_forms normalisées d'un concept.

        Utilisé par le coupe-circuit de la Brique 4 pour vérifier
        si le terme brut matche exactement une des formes du concept
        (canonical OU synonyme).

        Args:
            ontology_id: L'identifiant du concept (ex: "FIBRILLATION_ATRIALE").

        Returns:
            Set de strings normalisés (ex: {"fibrillation atriale", "af", "fa"}).
        """
        return self._normalized_forms_by_id.get(ontology_id, set())

    # ------------------------------------------------------------------
    # Recherche Dense (sémantique)
    # ------------------------------------------------------------------

    def _search_dense(
        self, query: str, pool_size: int = 30
    ) -> List[Tuple[int, float]]:
        """
        Recherche vectorielle : embedding de la query via OpenAI,
        puis similarité cosinus contre la matrice locale.

        Returns:
            Liste de (index, score) triée par score décroissant.
        """
        client = _get_client()

        response = client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=[query],
        )
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)

        # Dot product ≈ cosine similarity (vecteurs OpenAI déjà normalisés L2)
        similarities = self.embeddings @ query_vec

        top_indices = np.argsort(similarities)[::-1][:pool_size]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    # ------------------------------------------------------------------
    # Recherche Sparse (BM25)
    # ------------------------------------------------------------------

    def _search_sparse(
        self, query: str, pool_size: int = 30
    ) -> List[Tuple[int, float]]:
        """
        Recherche lexicale BM25 sur les surface_forms tokenisés.

        Returns:
            Liste de (index, score) triée par score décroissant.
        """
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:pool_size]

        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

    # ------------------------------------------------------------------
    # Fusion RRF
    # ------------------------------------------------------------------

    def _fuse_rrf(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int,
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion des résultats dense + sparse.

        Formule : RRF_score(doc) = Σ (boost / (RRF_K + rank))
        Le boost BM25 (1.5×) favorise les matchs lexicaux exacts (acronymes).

        Returns:
            Liste de (index, rrf_score) triée par score décroissant, taille k.
        """
        rrf_scores: Dict[int, float] = {}

        # Dense : poids standard (1.0)
        for rank, (idx, _sim) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.RRF_K + rank))

        # Sparse : boost si BM25 a un score > 0 (match lexical réel)
        for rank, (idx, bm25_score) in enumerate(sparse_results):
            boost = self.BM25_BOOST if bm25_score > 0 else 1.0
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (boost / (self.RRF_K + rank))

        # Tri final
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    # ------------------------------------------------------------------
    # Méthode publique : search_top_k
    # ------------------------------------------------------------------

    def search_top_k(
        self,
        query: str,
        k: int = 5,
        pool_factor: int = 3,
    ) -> List[Dict]:
        """
        Recherche hybride : trouve les Top-K concepts ontologiques
        les plus proches d'un terme brut.

        Le terme est normalisé via la même fonction que Brique 1
        avant d'être cherché dans les deux moteurs.

        Args:
            query:       Le terme brut extrait par GPT-4o (ex: "tachi supra").
            k:           Nombre de résultats à retourner.
            pool_factor: Facteur multiplicatif pour le pool de pré-sélection
                         (on cherche k × pool_factor dans chaque moteur avant fusion).

        Returns:
            Liste de dictionnaires, chacun contenant :
              - ontology_id   : identifiant du concept (ex: "FIBRILLATION_ATRIALE")
              - surface_form  : forme de surface matchée (ex: "FA")
              - concept_name  : nom canonique du concept
              - source_type   : "canonical" ou "synonym"
              - categorie     : catégorie ontologique
              - poids         : poids clinique (1-4)
              - rrf_score     : score de fusion RRF
              - is_exact_match: True si query normalisée == une surface_form normalisée du concept
        """
        query_norm = normalize_text(query)
        if not query_norm:
            return []

        pool_size = k * pool_factor

        # A. Dense (sémantique)
        dense_results = self._search_dense(query_norm, pool_size=pool_size)

        # B. Sparse (BM25)
        sparse_results = self._search_sparse(query_norm, pool_size=pool_size)

        # C. Fusion RRF
        fused = self._fuse_rrf(dense_results, sparse_results, k=k)

        # Index rapide des scores individuels par index de document
        dense_by_idx = {idx: score for idx, score in dense_results}
        sparse_by_idx = {idx: score for idx, score in sparse_results}

        # D. Formatage de la sortie
        results = []
        for idx, rrf_score in fused:
            doc = self.documents[idx]
            oid = doc["ontology_id"]
            # Vérifier si la query normalisée matche exactement une des
            # surface_forms du concept (canonical OU synonyme).
            # On teste aussi les variantes flexionnelles (pluriel/genre)
            # pour absorber "QRS larges" vs "QRS large", "atriale" vs "atrial".
            forms = self.get_all_normalized_forms(oid)
            exact = query_norm in forms or bool(_deflect(query_norm) & forms)
            results.append({
                "ontology_id": oid,
                "surface_form": doc["surface_form"],
                "concept_name": doc["concept_name"],
                "source_type": doc["source_type"],
                "categorie": doc["categorie"],
                "poids": doc["poids"],
                "rrf_score": round(rrf_score, 6),
                "cosine_score": round(dense_by_idx.get(idx, 0.0), 6),
                "bm25_score": round(sparse_by_idx.get(idx, 0.0), 4),
                "is_exact_match": exact,
            })

        return results

    # ------------------------------------------------------------------
    # Utilitaires de diagnostic
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Résumé lisible du moteur."""
        lines = [
            "=" * 60,
            "🔍 HYBRID SEARCH ENGINE — Résumé",
            "=" * 60,
            f"  Documents  : {len(self.documents)}",
            f"  Embeddings : {self.embeddings.shape}",
            f"  Modèle     : {self.index_meta.get('embedding_model', '?')}",
            f"  BM25       : {'✅' if self._bm25 is not None else '❌'}",
            f"  RRF K      : {self.RRF_K}",
            f"  BM25 Boost : {self.BM25_BOOST}×",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Point d'entrée CLI pour test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    index_dir = str(Path(__file__).parent / "rag_index")

    print(f"\n📂 Chargement de l'index depuis : {index_dir}\n")
    engine = HybridSearchEngine(index_dir)
    print(engine.describe())

    # Tests rapides
    test_queries = [
        "tachi supra",
        "FA",
        "bloc de branche gauche",
        "infarctus",
        "BBD",
        "fibrillation auriculaire",
        "onde T négative",
        "microvoltage",
    ]

    for query in test_queries:
        print(f"\n🔍 Query : \"{query}\"")
        results = engine.search_top_k(query, k=5)
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r['rrf_score']:.6f}] {r['ontology_id']} "
                f"— \"{r['surface_form']}\" ({r['source_type']}, {r['categorie']})"
            )

    print("\n✅ Brique 3 — Recherche Hybride terminée.")
