import numpy as np
import torch

from langchain_huggingface import HuggingFaceEmbeddings

from simple_rag.config import EMBEDDING_MODEL, HF_TOKEN 


class Embedder:

    def __init__(self):

        kwargs = {}
        # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone
        kwargs['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        if HF_TOKEN:
            kwargs['token'] = HF_TOKEN

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=kwargs,
            encode_kwargs={'normalize_embeddings': False}
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(self.embeddings.embed_documents(texts))
