"""
🧱 Brique 1 — Socle Symbolique & Vectoriel
============================================
Script "one-off" : transforme l'ontologie OWL (JSON) en base vectorielle locale.

Objectif : prendre ontology_from_owl.json, normaliser le texte, générer les
embeddings via OpenAI text-embedding-3-small (1536 dims), et sauvegarder en local
(matrice NumPy + registre JSON) pour recherche de similarité instantanée en RAM.

Chaque concept génère N "documents" indexés :
  - Le nom canonique (concept_name)
  - Chaque synonyme
  - PAS les implications (réservées au Juge Neurosymbolique, Brique 4)

Sortie :
  vecteurs_ontologie.npy   — matrice N×1536 (float32)
  metadata_ontologie.json  — registre {index i ↔ ligne i de la matrice}
  + BM25 en bonus pour recherche hybride

Auteur : BMad Team
Date   : 2026-02-25
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass : un « document » indexé pointant vers un concept ontologique
# ---------------------------------------------------------------------------

@dataclass
class OntologyDocument:
    """Un terme indexable lié à un concept de l'ontologie."""
    ontology_id: str          # ex: "FIBRILLATION_ATRIALE"
    surface_form: str         # ex: "FA" ou "Fibrillation atriale"
    source_type: str          # "canonical" | "synonym" | "implication"
    concept_name: str         # Nom canonique du concept
    categorie: str            # "DIAGNOSTIC_URGENT", etc.
    poids: int                # 1-4
    

# ---------------------------------------------------------------------------
# Fonctions utilitaires de normalisation textuelle
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalisation stricte pour la recherche hybride.
    
    Règles métier :
    1. Passage en minuscules
    2. Suppression totale des accents (décomposition NFD)
    3. Remplacement de la ponctuation (. - _ ) par des espaces
    4. Suppression des espaces multiples
    """
    # 1) Minuscules
    text = text.lower().strip()
    # 2) NFD : décomposer les caractères accentués, puis retirer les diacritiques
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # 3) Ponctuation → espaces
    text = re.sub(r'[.\-_]+', ' ', text)
    # 4) Espaces multiples → un seul
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenisation simple pour BM25 (split sur espaces/ponctuations)."""
    import re
    normalized = normalize_text(text)
    tokens = re.split(r'[\s\-_/\'\",.;:()]+', normalized)
    return [t for t in tokens if len(t) > 1]  # filtre les tokens d'1 char


# ---------------------------------------------------------------------------
# Classe principale : OntologyIndex
# ---------------------------------------------------------------------------

class OntologyIndex:
    """
    Index dual (vectoriel + BM25) sur les concepts de l'ontologie ECG.
    
    Usage:
        idx = OntologyIndex(ontology_path="data/ontology_from_owl.json")
        idx.build(include_implications=False)
        idx.save("rag_index/")
        
        # Plus tard :
        idx2 = OntologyIndex.load("rag_index/")
        results = idx2.search("fibrillation auriculaire", top_k=5)
    """
    
    # Modèle OpenAI pour les embeddings
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMS = 1536
    EMBEDDING_BATCH_SIZE = 512  # limite OpenAI : 2048 inputs par requête

    def __init__(self, ontology_path: Optional[str] = None):
        self.ontology_path = ontology_path
        self.documents: List[OntologyDocument] = []
        
        # Index BM25
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[List[str]] = []
        
        # Index vectoriel (matrice N×1536, float32)
        self._embeddings: Optional[np.ndarray] = None
        
        # Client OpenAI (initialisé à la demande)
        self._client: Optional[OpenAI] = None
        
        # Métadonnées
        self.metadata: Dict = {}

    def _get_client(self) -> OpenAI:
        """Retourne le client OpenAI, en le créant si nécessaire."""
        if self._client is None:
            # Chercher le .env dans le dossier courant ou dans ECG lecture (projet parent)
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
                load_dotenv()  # fallback : cherche automatiquement
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY non trouvée. "
                    "Ajoutez-la dans un fichier .env ou en variable d'environnement."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    # ------------------------------------------------------------------
    # Étape 1 : Parsing de l'ontologie JSON
    # ------------------------------------------------------------------
    
    def _parse_ontology(self, include_implications: bool = False) -> List[OntologyDocument]:
        """
        Parse le JSON de l'ontologie et génère la liste de documents indexables.
        
        Structure attendue du JSON (section concept_mappings) :
        {
            "FIBRILLATION_ATRIALE": {
                "concept_name": "Fibrillation atriale",
                "synonymes": ["AF", "FA"],
                "implications": ["Absence d'onde P", ...],
                "poids": 3,
                "categorie": "DIAGNOSTIC_MAJEUR"
            },
            ...
        }
        """
        path = Path(self.ontology_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontologie introuvable : {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        concept_mappings = data.get("concept_mappings", {})
        if not concept_mappings:
            # Fallback: ontology V2 uses "concepts" key
            concept_mappings = data.get("concepts", {})
        if not concept_mappings:
            raise ValueError("Clé 'concept_mappings' (ou 'concepts') absente ou vide dans l'ontologie.")
        
        documents = []
        stats = {"canonical": 0, "synonym": 0, "implication": 0, "skipped_empty": 0}
        
        for ontology_id, concept_data in concept_mappings.items():
            concept_name = concept_data.get("concept_name", ontology_id)
            categorie = concept_data.get("categorie", "UNKNOWN")
            poids = concept_data.get("poids", 1)
            synonymes = concept_data.get("synonymes", [])
            implications = concept_data.get("implications", [])
            
            # 1) Nom canonique — toujours indexé
            documents.append(OntologyDocument(
                ontology_id=ontology_id,
                surface_form=concept_name,
                source_type="canonical",
                concept_name=concept_name,
                categorie=categorie,
                poids=poids,
            ))
            stats["canonical"] += 1
            
            # 2) Synonymes
            for syn in synonymes:
                syn_clean = syn.strip()
                if not syn_clean:
                    stats["skipped_empty"] += 1
                    continue
                documents.append(OntologyDocument(
                    ontology_id=ontology_id,
                    surface_form=syn_clean,
                    source_type="synonym",
                    concept_name=concept_name,
                    categorie=categorie,
                    poids=poids,
                ))
                stats["synonym"] += 1
            
            # 3) Implications (optionnel — désactivé par défaut)
            if include_implications:
                for impl in implications:
                    impl_clean = impl.strip()
                    if not impl_clean:
                        stats["skipped_empty"] += 1
                        continue
                    documents.append(OntologyDocument(
                        ontology_id=ontology_id,
                        surface_form=impl_clean,
                        source_type="implication",
                        concept_name=concept_name,
                        categorie=categorie,
                        poids=poids,
                    ))
                    stats["implication"] += 1
        
        logger.info(f"📋 Parsing ontologie : {len(documents)} documents générés")
        logger.info(f"   Canoniques: {stats['canonical']}, Synonymes: {stats['synonym']}, "
                     f"Implications: {stats['implication']}, Skipped: {stats['skipped_empty']}")
        
        self.metadata = {
            "total_concepts": len(concept_mappings),
            "total_documents": len(documents),
            "stats": stats,
            "include_implications": include_implications,
            "source_file": str(path.name),
        }
        
        return documents
    
    # ------------------------------------------------------------------
    # Étape 2 : Construction de l'index BM25 (lexical)
    # ------------------------------------------------------------------
    
    def _build_bm25(self):
        """Construit l'index BM25 sur les surface_forms tokenisés."""
        t0 = time.time()
        self._bm25_corpus = [tokenize(doc.surface_form) for doc in self.documents]
        self._bm25 = BM25Okapi(self._bm25_corpus)
        elapsed = time.time() - t0
        logger.info(f"🔤 Index BM25 construit en {elapsed:.2f}s ({len(self._bm25_corpus)} documents)")
    
    # ------------------------------------------------------------------
    # Étape 3 : Construction de l'index vectoriel (embeddings)
    # ------------------------------------------------------------------
    
    def _build_embeddings(self):
        """
        Encode tous les surface_forms via OpenAI text-embedding-3-small.
        
        Gère le batching automatique (max EMBEDDING_BATCH_SIZE par requête).
        Les embeddings sont normalisés L2 par l'API OpenAI.
        """
        client = self._get_client()
        
        surface_forms = [doc.surface_form for doc in self.documents]
        n = len(surface_forms)
        all_embeddings = np.zeros((n, self.EMBEDDING_DIMS), dtype=np.float32)
        
        t0 = time.time()
        for start in range(0, n, self.EMBEDDING_BATCH_SIZE):
            end = min(start + self.EMBEDDING_BATCH_SIZE, n)
            batch = surface_forms[start:end]
            
            response = client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=batch,
            )
            
            for item in response.data:
                all_embeddings[start + item.index] = item.embedding
            
            logger.info(f"   📡 Batch [{start}:{end}] embeddings reçus")
        
        self._embeddings = all_embeddings
        elapsed = time.time() - t0
        logger.info(f"🧠 Embeddings calculés en {elapsed:.2f}s "
                     f"(shape: {self._embeddings.shape}, modèle: {self.EMBEDDING_MODEL})")
    
    # ------------------------------------------------------------------
    # Build complet
    # ------------------------------------------------------------------
    
    def build(self, include_implications: bool = False):
        """
        Pipeline complet : parse → BM25 → embeddings.
        
        Args:
            include_implications: Si True, indexe aussi les implications
                                  (termes enfants) comme documents séparés.
                                  ⚠️ Risque de bruit, à tester.
        """
        logger.info("=" * 60)
        logger.info("🔨 CONSTRUCTION DE L'INDEX ONTOLOGIQUE")
        logger.info("=" * 60)
        
        # 1) Parse
        self.documents = self._parse_ontology(include_implications=include_implications)
        
        # 2) BM25
        self._build_bm25()
        
        # 3) Embeddings
        self._build_embeddings()
        
        logger.info("=" * 60)
        logger.info(f"✅ Index prêt : {len(self.documents)} documents, "
                     f"{self.metadata['total_concepts']} concepts")
        logger.info("=" * 60)
        
        return self
    
    # ------------------------------------------------------------------
    # Recherche hybride (sera utilisée à la Brique 3)
    # ------------------------------------------------------------------
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[OntologyDocument, float]]:
        """Recherche lexicale BM25."""
        if self._bm25 is None:
            raise RuntimeError("Index BM25 non construit. Appelez build() d'abord.")
        
        tokens = tokenize(query)
        if not tokens:
            return []
        
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def search_vector(self, query: str, top_k: int = 10) -> List[Tuple[OntologyDocument, float]]:
        """Recherche vectorielle par similarité cosinus (embedding via OpenAI)."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings non construits. Appelez build() d'abord.")
        
        client = self._get_client()
        
        response = client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=[query],
        )
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Dot product ≈ cosine similarity (embeddings normalisés par l'API)
        similarities = self._embeddings @ query_embedding
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> List[Tuple[OntologyDocument, float]]:
        """
        Recherche hybride : fusion des scores BM25 et vectoriel.
        
        Stratégie : Reciprocal Rank Fusion (RRF) pondérée.
        Score_final(doc) = bm25_weight * (1 / (k + rank_bm25)) 
                         + vector_weight * (1 / (k + rank_vector))
        avec k = 60 (constante RRF standard).
        """
        k = 60  # constante RRF
        pool_size = top_k * 3  # chercher plus large pour la fusion
        
        # Récupérer les deux listes
        bm25_results = self.search_bm25(query, top_k=pool_size)
        vector_results = self.search_vector(query, top_k=pool_size)
        
        # Construire les maps rank par ontology_id + surface_form
        def result_key(doc: OntologyDocument) -> str:
            return f"{doc.ontology_id}|{doc.surface_form}"
        
        bm25_ranks = {}
        for rank, (doc, _score) in enumerate(bm25_results):
            key = result_key(doc)
            if key not in bm25_ranks:
                bm25_ranks[key] = rank + 1
        
        vector_ranks = {}
        for rank, (doc, _score) in enumerate(vector_results):
            key = result_key(doc)
            if key not in vector_ranks:
                vector_ranks[key] = rank + 1
        
        # Union de tous les documents candidats
        all_docs = {}
        for doc, _ in bm25_results + vector_results:
            key = result_key(doc)
            if key not in all_docs:
                all_docs[key] = doc
        
        # Calcul du score RRF
        scored = []
        for key, doc in all_docs.items():
            rrf_bm25 = bm25_weight * (1.0 / (k + bm25_ranks.get(key, pool_size + 1)))
            rrf_vector = vector_weight * (1.0 / (k + vector_ranks.get(key, pool_size + 1)))
            scored.append((doc, rrf_bm25 + rrf_vector))
        
        # Trier par score décroissant
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Dédupliquer par ontology_id (garder le meilleur surface_form par concept)
        seen_ids = set()
        deduped = []
        for doc, score in scored:
            if doc.ontology_id not in seen_ids:
                seen_ids.add(doc.ontology_id)
                deduped.append((doc, score))
            if len(deduped) >= top_k:
                break
        
        return deduped
    
    # ------------------------------------------------------------------
    # Sauvegarde / Chargement (sans le modèle, juste les données)
    # ------------------------------------------------------------------
    
    def save(self, directory: str):
        """
        Sauvegarde l'index sur disque.
        
        Fichiers produits :
        - vecteurs_ontologie.npy   : matrice N×1536 (float32)
        - metadata_ontologie.json  : registre {index i ↔ ligne i} + métadonnées
        - bm25_corpus.json         : corpus tokenisé pour reconstruire BM25
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Registre métadonnées (index i ↔ document i)
        docs_data = [
            {
                "ontology_id": d.ontology_id,
                "surface_form": d.surface_form,
                "source_type": d.source_type,
                "concept_name": d.concept_name,
                "categorie": d.categorie,
                "poids": d.poids,
            }
            for d in self.documents
        ]
        meta_output = {
            "documents": docs_data,
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dims": self.EMBEDDING_DIMS,
            **self.metadata,
        }
        with open(out_dir / "metadata_ontologie.json", 'w', encoding='utf-8') as f:
            json.dump(meta_output, f, ensure_ascii=False, indent=2)
        
        # Matrice d'embeddings
        if self._embeddings is not None:
            np.save(out_dir / "vecteurs_ontologie.npy", self._embeddings)
        
        # BM25 corpus (pour reconstruire l'index au chargement)
        with open(out_dir / "bm25_corpus.json", 'w', encoding='utf-8') as f:
            json.dump(self._bm25_corpus, f, ensure_ascii=False)
        
        logger.info(f"💾 Index sauvegardé dans {out_dir}/ "
                     f"(vecteurs_ontologie.npy + metadata_ontologie.json)")
    
    @classmethod
    def load(cls, directory: str) -> "OntologyIndex":
        """
        Charge un index sauvegardé depuis le disque.
        
        Fichiers attendus :
        - vecteurs_ontologie.npy
        - metadata_ontologie.json
        - bm25_corpus.json
        """
        in_dir = Path(directory)
        if not in_dir.exists():
            raise FileNotFoundError(f"Répertoire d'index introuvable : {in_dir}")
        
        idx = cls()
        
        # Métadonnées + documents
        meta_path = in_dir / "metadata_ontologie.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata_ontologie.json introuvable dans {in_dir}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        docs_data = meta.pop("documents", [])
        idx.documents = [OntologyDocument(**d) for d in docs_data]
        idx.metadata = meta
        
        # Embeddings
        emb_path = in_dir / "vecteurs_ontologie.npy"
        if emb_path.exists():
            idx._embeddings = np.load(emb_path)
        
        # BM25
        bm25_path = in_dir / "bm25_corpus.json"
        if bm25_path.exists():
            with open(bm25_path, 'r', encoding='utf-8') as f:
                idx._bm25_corpus = json.load(f)
            idx._bm25 = BM25Okapi(idx._bm25_corpus)
        
        logger.info(f"📂 Index chargé : {len(idx.documents)} documents depuis {in_dir}/")
        return idx
    
    # ------------------------------------------------------------------
    # Utilitaires de diagnostic
    # ------------------------------------------------------------------
    
    def describe(self) -> str:
        """Résumé lisible de l'index."""
        lines = [
            "=" * 60,
            "📊 ONTOLOGY INDEX — Résumé",
            "=" * 60,
            f"  Concepts   : {self.metadata.get('total_concepts', '?')}",
            f"  Documents  : {len(self.documents)}",
            f"    ├─ Canoniques   : {self.metadata.get('stats', {}).get('canonical', '?')}",
            f"    ├─ Synonymes    : {self.metadata.get('stats', {}).get('synonym', '?')}",
            f"    └─ Implications : {self.metadata.get('stats', {}).get('implication', '?')}",
            f"  Modèle     : {self.EMBEDDING_MODEL} ({self.EMBEDDING_DIMS} dims)",
            f"  Embeddings : {'✅' if self._embeddings is not None else '❌'} "
            f"{'(' + str(self._embeddings.shape) + ')' if self._embeddings is not None else ''}",
            f"  BM25       : {'✅' if self._bm25 is not None else '❌'}",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def get_concept_by_id(self, ontology_id: str) -> List[OntologyDocument]:
        """Retourne tous les documents liés à un ontology_id."""
        return [d for d in self.documents if d.ontology_id == ontology_id]


# ---------------------------------------------------------------------------
# Point d'entrée CLI pour construction de l'index
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    # Chemins par défaut
    ontology_path = str(Path(__file__).parent.parent / "ECG lecture" / "data" / "ontology_v2.json")
    index_dir = str(Path(__file__).parent / "rag_index")
    
    print(f"\n📥 Ontologie : {ontology_path}")
    print(f"📤 Index     : {index_dir}\n")
    
    # Build
    idx = OntologyIndex(ontology_path=ontology_path)
    idx.build(include_implications=False)
    
    # Describe
    print(idx.describe())
    
    # Save
    idx.save(index_dir)
    
    # Quick sanity check
    print("\n🔍 Test rapide — Recherche hybride pour 'FA' :")
    results = idx.search_hybrid("FA", top_k=5)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.ontology_id} — '{doc.surface_form}' ({doc.source_type})")
    
    print("\n🔍 Test rapide — Recherche hybride pour 'bloc de branche gauche' :")
    results = idx.search_hybrid("bloc de branche gauche", top_k=5)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.ontology_id} — '{doc.surface_form}' ({doc.source_type})")
    
    print("\n🔍 Test rapide — Recherche hybride pour 'infarctus' :")
    results = idx.search_hybrid("infarctus", top_k=5)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.ontology_id} — '{doc.surface_form}' ({doc.source_type})")
    
    print("\n✅ Brique 1 terminée.")
