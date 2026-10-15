"""
Service d'embedding d'images.

L'ancienne version de ce fichier était un copier-coller quasi-identique de
embedding_service.py (même classe, même logique texte) — malgré son nom,
il ne contenait aucune ligne concernant des images. Le vrai code CLIP
vivait en dur dans rag_service.py.

Cette version fait ce que le nom du fichier promet : un vrai service
d'embedding d'images, utilisé par rag_service.py à la place du code inline.

Utilise OpenCLIP (checkpoints LAION) plutôt que le CLIP original d'OpenAI
(2021) : à dimension de sortie égale (512, compatible avec les collections
Qdrant existantes), OpenCLIP surclasse le CLIP original sur la recherche
cross-modale. Nécessite le paquet `open-clip-torch` (à ajouter à
requirements.txt).
"""
import logging
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ImageEmbeddingService:

    # (model_name, pretrained) -> dimension de sortie
    SUPPORTED_MODELS: ClassVar[dict[tuple[str, str], int]] = {
        ("ViT-B-32", "laion2b_s34b_b79k"): 512,   # Bon compromis vitesse/qualité, défaut
        ("ViT-B-16", "laion2b_s34b_b88k"): 512,
        ("ViT-L-14", "laion2b_s32b_b82k"): 768,
    }

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading image model: {model_name} ({pretrained})")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.model.eval()
            self.dimension = self.SUPPORTED_MODELS.get((model_name, pretrained), 512)
            self.tokenizer = open_clip.get_tokenizer(model_name)
            logger.info(f"Image model loaded: dim={self.dimension}, device={self.device}")
        except Exception as e:
            logger.error(f"Error loading image model {model_name}/{pretrained}: {e}")
            if (model_name, pretrained) != ("ViT-B-32", "laion2b_s34b_b79k"):
                logger.info("Fallback to ViT-B-32 / laion2b_s34b_b79k")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
                )
                self.model.eval()
                self.model_name = "ViT-B-32"
                self.pretrained = "laion2b_s34b_b79k"
                self.dimension = 512
                self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            else:
                raise

    def embed_image(self, image: bytes | BytesIO | Path | str) -> list[float]:
        """Encode une image (bytes, BytesIO, ou chemin) en vecteur normalisé."""
        try:
            if isinstance(image, bytes):
                img = Image.open(BytesIO(image)).convert("RGB")
            elif isinstance(image, BytesIO):
                img = Image.open(image).convert("RGB")
            else:
                img = Image.open(image).convert("RGB")

            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding_list = embedding.cpu().numpy().flatten().tolist()
            return embedding_list

        except Exception as e:
            logger.error(f"Image embedding error: {e}")
            return [0.0] * self.dimension

    def embed_text_query(self, text: str) -> list[float]:
        """
        Encodes a text query into the SAME joint embedding space as
        embed_image() — OpenCLIP's text encoder, not
        IndustrialEmbeddingService's sentence-transformers space.

        Regression fix: rag_service.py used to query the image collection
        with the sentence-transformers query vector (384-dim) against a
        512-dim CLIP collection — a dimension mismatch, and even where
        dimensions happened to match it would still be semantically
        meaningless (different embedding spaces entirely). Use this method
        for any query against the image collection.
        """
        try:
            tokens = self.tokenizer([text]).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_text(tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding_list = embedding.cpu().numpy().flatten().tolist()
            return embedding_list
        except Exception as e:
            logger.error(f"Text query embedding error: {e}")
            return [0.0] * self.dimension

    def embed_images_batch(self, images: list[bytes | BytesIO | Path | str]) -> list[list[float]]:
        return [self.embed_image(img) for img in images]

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "dimension": self.dimension,
            "device": self.device,
        }

    @classmethod
    def get_available_models(cls) -> list[str]:
        return [f"{name}/{pretrained}" for name, pretrained in cls.SUPPORTED_MODELS]