"""
Store vectoriel Qdrant avec hybrid search réel.

L'ancienne version définissait `search_hybrid` DEUX FOIS dans la même classe.
La deuxième définition (plus bas dans le fichier) écrasait silencieusement la
première en Python — et aucune des deux ne faisait de vrai hybrid search :
la première contenait littéralement `# TODO: Implémenter la vraie recherche
hybride avec BM25`, l'autre était un simple alias vers search_vector.

Cette version implémente un vrai hybrid search dense+sparse avec fusion RRF,
nativement supporté par Qdrant (Query API, disponible depuis qdrant-client
>= 1.10) :
  - vecteur dense : embeddings sentence-transformers (sémantique)
  - vecteur sparse : BM25 via fastembed (mots-clés exacts, gratuit, pas de
    dépendance à un service externe)
  - fusion : Reciprocal Rank Fusion (RRF) côté Qdrant

Nécessite d'ajouter `fastembed` à requirements.txt.

⚠️ Migration : passer d'un vecteur non-nommé à des vecteurs nommés (dense +
sparse) change le schéma de la collection Qdrant. Les documents déjà
indexés doivent être ré-ingérés après ce changement.
"""
import logging
import uuid
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class SearchResult:
    """Résultat de recherche avec métadonnées"""

    def __init__(self, id: str, score: float, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.score = score
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata
        }


class IndustrialVectorStore:
    """Service de stockage vectoriel avec Qdrant. Supporte le hybrid search réel (dense + sparse BM25)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: str = "documents",
        vector_dim: int = 384,
        use_hybrid_search: bool = True
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.use_hybrid_search = use_hybrid_search
        self._sparse_model = None  # chargé paresseusement, seulement si use_hybrid_search

        try:
            if api_key:
                self.client = QdrantClient(host=host, port=port, api_key=api_key)
            else:
                self.client = QdrantClient(host=host, port=port)
            logger.info(f"✅ Connecté à Qdrant: {host}:{port}")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Qdrant: {e}")
            logger.info("Fallback vers Qdrant en mémoire")
            self.client = QdrantClient(":memory:")

        self._ensure_collection()

    def _get_sparse_model(self):
        """Charge fastembed BM25 à la demande (évite le coût au démarrage si hybrid désactivé)."""
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            logger.info("Loading sparse model: Qdrant/bm25")
            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._sparse_model

    def _build_filter(self, filter_conditions: Optional[Dict]) -> Optional[Filter]:
        if not filter_conditions:
            return None
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filter_conditions.items()
        ]
        return Filter(must=conditions) if conditions else None

    def _ensure_collection(self) -> None:
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"📦 Création de la collection: {self.collection_name}")

                sparse_config = None
                if self.use_hybrid_search:
                    sparse_config = {
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    }

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=self.vector_dim,
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config=sparse_config,
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=10000,
                        memmap_threshold=10000
                    ),
                )

                for field in ["filename", "chunk_index", "source"]:
                    try:
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field,
                            field_schema="keyword"
                        )
                    except Exception:
                        pass

                logger.info(f"✅ Collection créée: {self.collection_name}")
            else:
                logger.info(f"✅ Collection existante: {self.collection_name}")

        except Exception as e:
            logger.warning(f"⚠️ Erreur collection: {e}")

    def upsert_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        if not chunks:
            return 0

        if len(chunks) != len(embeddings) or len(chunks) != len(metadata):
            raise ValueError("chunks, embeddings et metadata doivent avoir la même longueur")

        truncated_chunks = [c[:5000] for c in chunks]

        sparse_embeddings = None
        if self.use_hybrid_search:
            try:
                sparse_model = self._get_sparse_model()
                sparse_embeddings = list(sparse_model.embed(truncated_chunks))
            except Exception as e:
                logger.warning(f"Sparse embedding failed, indexing dense-only for this batch: {e}")

        points = []
        for i in range(len(chunks)):
            point_id = str(uuid.uuid4())
            vector: Dict[str, Any] = {DENSE_VECTOR_NAME: embeddings[i]}

            if sparse_embeddings is not None:
                se = sparse_embeddings[i]
                vector[SPARSE_VECTOR_NAME] = models.SparseVector(
                    indices=se.indices.tolist(),
                    values=se.values.tolist()
                )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": truncated_chunks[i],
                        **metadata[i]
                    }
                )
            )

        total_inserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                total_inserted += len(batch)
            except Exception as e:
                logger.error(f"Erreur insertion batch {i}: {e}")

        logger.info(f"💾 {total_inserted} points insérés")
        return total_inserted

    def search_vector(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Recherche vectorielle dense pure."""
        qdrant_filter = self._build_filter(filter_conditions)

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True
            )

            return [
                SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    content=hit.payload.get("content", ""),
                    metadata={k: v for k, v in hit.payload.items() if k != "content"}
                )
                for hit in results.points
            ]

        except Exception as e:
            logger.error(f"Erreur recherche vectorielle: {e}")
            return []

    def search_hybrid(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Recherche hybride réelle : fusion RRF entre recherche dense (sémantique)
        et recherche sparse BM25 (mots-clés exacts) — les deux exécutées côté
        Qdrant en un seul appel réseau via prefetch + FusionQuery.

        Note: le score renvoyé est un score de fusion RRF, pas une similarité
        cosinus 0-1. score_threshold n'est donc pas appliqué ici (l'échelle
        n'est pas comparable) ; filtrer côté appelant si besoin.
        """
        if not self.use_hybrid_search:
            return self.search_vector(query_vector, limit, score_threshold, filter_conditions)

        qdrant_filter = self._build_filter(filter_conditions)

        try:
            sparse_model = self._get_sparse_model()
            sparse_query = next(sparse_model.embed([query_text]))
            sparse_vector = models.SparseVector(
                indices=sparse_query.indices.tolist(),
                values=sparse_query.values.tolist()
            )

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using=DENSE_VECTOR_NAME,
                        filter=qdrant_filter,
                        limit=limit * 2
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        filter=qdrant_filter,
                        limit=limit * 2
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True
            )

            return [
                SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    content=hit.payload.get("content", ""),
                    metadata={k: v for k, v in hit.payload.items() if k != "content"}
                )
                for hit in results.points
            ]

        except Exception as e:
            logger.error(f"Erreur recherche hybride, fallback vers recherche dense: {e}")
            return self.search_vector(query_vector, limit, score_threshold, filter_conditions)

    def delete_documents(
        self,
        filter_conditions: Optional[Dict] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Supprime des documents. Retourne le nombre réel de points supprimés
        (l'ancienne version retournait toujours 0 pour une suppression par
        filtre — "compte approximatif" jamais implémenté).
        """
        try:
            if ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=ids)
                )
                return len(ids)

            elif filter_conditions:
                qdrant_filter = self._build_filter(filter_conditions)

                matched_ids = []
                next_offset = None
                while True:
                    batch, next_offset = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=qdrant_filter,
                        limit=1000,
                        offset=next_offset,
                        with_payload=False
                    )
                    matched_ids.extend([p.id for p in batch])
                    if next_offset is None:
                        break

                if matched_ids:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=models.PointIdsList(points=matched_ids)
                    )
                return len(matched_ids)

            return 0

        except Exception as e:
            logger.error(f"Erreur suppression: {e}")
            return 0

    def get_documents_by_filename(self, filename: str) -> List[SearchResult]:
        """
        Récupère tous les chunks d'un fichier. Utilise scroll+filter plutôt
        qu'une recherche vectorielle avec un vecteur factice [0.0, ...]
        (ancienne approche : lente et sémantiquement incorrecte pour un
        simple filtrage exact).
        """
        try:
            points = []
            next_offset = None
            qdrant_filter = self._build_filter({"filename": filename})

            while True:
                batch, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_filter,
                    limit=1000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False
                )
                points.extend(batch)
                if next_offset is None:
                    break

            return [
                SearchResult(
                    id=str(p.id),
                    score=1.0,
                    content=p.payload.get("content", ""),
                    metadata={k: v for k, v in p.payload.items() if k != "content"}
                )
                for p in points
            ]
        except Exception as e:
            logger.error(f"Erreur récupération par filename: {e}")
            return []

    def get_all_filenames(self) -> List[str]:
        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )
            filenames = set()
            for point in scroll_result[0]:
                if "filename" in point.payload:
                    filenames.add(point.payload["filename"])
            return sorted(list(filenames))
        except Exception as e:
            logger.error(f"Erreur récupération fichiers: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "dimension": self.vector_dim,
                "hybrid_search_enabled": self.use_hybrid_search
            }
        except Exception as e:
            return {"error": str(e), "name": self.collection_name}

    def is_healthy(self) -> bool:
        """
        Vérifie une connectivité RÉELLE à Qdrant via un appel réseau, pas
        juste que self.client existe en tant qu'objet Python. L'ancienne
        approche (`rag_service.vector_store is not None`) restait vraie
        même Qdrant complètement injoignable — __init__ et _ensure_collection
        avalent les erreurs de connexion sans jamais lever d'exception.
        """
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection supprimée: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Erreur suppression collection: {e}")
            return False

    def clear_collection(self) -> int:
        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=False
            )
            ids = [point.id for point in scroll_result[0]]
            if ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=ids)
                )
            return len(ids)
        except Exception as e:
            logger.error(f"Erreur vidage collection: {e}")
            return 0