import torch
import torch.nn as nn
from .base import BaseKTModel


class DummyKTModel(BaseKTModel):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_dim = cfg["model"]["embedding_dim"]
        self.q_emb = nn.Embedding(num_embeddings=100, embedding_dim=self.embedding_dim)
        self.c_emb = nn.Embedding(num_embeddings=20, embedding_dim=self.embedding_dim)
        self.linear = nn.Linear(self.embedding_dim * 2, 1)

    def forward(self, question_seq, concept_seq, response_seq, seq_len):
        q = self.q_emb(question_seq)
        c = self.c_emb(concept_seq)
        x = torch.cat([q, c], dim=-1)
        logits = self.linear(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs
