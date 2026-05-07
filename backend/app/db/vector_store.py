import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Résultat de recherche avec métadonnées"""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata
        }


class IndustrialVectorStore:
    """
    Service de stockage vectoriel avec Qdrant.
    Supporte la recherche hybride (vectorielle + plein texte).
    Version finale - ne plus modifier.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: str = "documents",
        vector_dim: int = 384,
        use_hybrid_search: bool = True
    ):
        """
        Args:
            host: Hôte Qdrant
            port: Port Qdrant
            api_key: Clé API Qdrant Cloud (optionnel)
            collection_name: Nom de la collection
            vector_dim: Dimension des vecteurs d'embedding
            use_hybrid_search: Activer la recherche hybride (vector + texte)
        """
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.use_hybrid_search = use_hybrid_search
        
        # Connexion à Qdrant
        try:
            if api_key:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key
                )
            else:
                self.client = QdrantClient(host=host, port=port)
            
            logger.info(f"✅ Connecté à Qdrant: {host}:{port}")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Qdrant: {e}")
            # Fallback en mémoire
            logger.info("Fallback vers Qdrant en mémoire")
            self.client = QdrantClient(":memory:")
        
        # Initialisation de la collection
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Crée la collection si elle n'existe pas"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                logger.info(f"📦 Création de la collection: {self.collection_name}")
                
                # Configuration de base
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=10000,
                        memmap_threshold=10000
                    ),
                )
                
                # Index pour la recherche hybride (texte)
                if self.use_hybrid_search:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="content",
                        field_schema=models.TextIndexParams(
                            type="text",
                            tokenizer=models.TokenizerType.WORD,
                            min_token_len=2,
                            max_token_len=20,
                            lowercase=True
                        ),
                    )
                
                # Index pour les métadonnées fréquentes
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
        """
        Insère des documents dans la base vectorielle.
        
        Args:
            chunks: Liste des textes
            embeddings: Liste des vecteurs correspondants
            metadata: Liste des métadonnées
            batch_size: Taille des lots pour l'insertion
        
        Returns:
            Nombre de points insérés
        """
        if not chunks:
            return 0
        
        if len(chunks) != len(embeddings) or len(chunks) != len(metadata):
            raise ValueError("chunks, embeddings et metadata doivent avoir la même longueur")
        
        points = []
        for i in range(len(chunks)):
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload={
                        "content": chunks[i][:5000],  # Limite pour éviter payload trop gros
                        **metadata[i]
                    }
                )
            )
        
        # Insertion par lots
        total_inserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
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
        """
        Recherche vectorielle pure.
        
        Args:
            query_vector: Vecteur de la requête
            limit: Nombre de résultats
            score_threshold: Seuil de similarité minimum (0-1)
            filter_conditions: Filtres sur les métadonnées
        
        Returns:
            Liste des résultats
        """
        # Construction du filtre
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
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
        vector_weight: float = 0.7,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Recherche hybride (vectorielle + texte).
        Pour l'instant, utilise la recherche vectorielle.
        TODO: Implémenter la vraie recherche hybride avec BM25.
        """
        return self.search_vector(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )    
    def delete_documents(
        self,
        filter_conditions: Optional[Dict] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Supprime des documents de la collection.
        
        Args:
            filter_conditions: Filtres sur les métadonnées
            ids: IDs spécifiques à supprimer
        
        Returns:
            Nombre de documents supprimés
        """
        try:
            if ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=ids)
                )
                return len(ids)
            
            elif filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=models.FilterSelector(
                            filter=Filter(must=conditions)
                        )
                    )
                    # Compter approximatif
                    return 0
            return 0
            
        except Exception as e:
            logger.error(f"Erreur suppression: {e}")
            return 0
    
    def get_documents_by_filename(self, filename: str) -> List[SearchResult]:
        """Récupère tous les chunks d'un fichier spécifique"""
        return self.search_vector(
            query_vector=[0.0] * self.vector_dim,  # Dummy vector
            limit=1000,
            filter_conditions={"filename": filename}
        )
    
    def get_all_filenames(self) -> List[str]:
        """Récupère la liste de tous les noms de fichiers indexés"""
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
        """Retourne les statistiques de la collection"""
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
    
    def delete_collection(self) -> bool:
        """Supprime toute la collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection supprimée: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Erreur suppression collection: {e}")
            return False
    def search_hybrid(self, query_vector: List[float], query_text: str, limit: int = 10, **kwargs):
        return self.search_vector(query_vector, limit=limit, **kwargs)
    def clear_collection(self) -> int:
        """Vide la collection sans la supprimer"""
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