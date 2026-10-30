import torch
import torch.nn as nn

class BaseWorldModel(nn.Module):
    """
    基类，强制预留action入参
    公式：s_{t+1}=f(s_t, a_t)
    学生B(DKT/AKT)、学生C(GRU world model)的模型都继承这个类
    """
    def forward(self, state, action):
        """
        :param state: 当前学生状态，shape [B, seq_len]
        :param action: 题目/知识点/提示/教学干预，shape [B, seq_len]
        :return next_state
        """
        raise NotImplementedError("请子类实现forward")


class DummyKTModel(BaseWorldModel):
    """仅用于smoke test的假模型，无真实效果"""
    def __init__(self, cfg):
        super().__init__()
        self.embedding_dim = cfg["model"]["embedding_dim"]
        self.hidden_dim = cfg["model"]["hidden_dim"]
        # 把离散id映射成embedding
        self.state_emb = nn.Embedding(num_embeddings=10, embedding_dim=self.embedding_dim)
        self.action_emb = nn.Embedding(num_embeddings=10, embedding_dim=self.embedding_dim)
        self.linear = nn.Linear(self.embedding_dim * 2, self.hidden_dim)

    def forward(self, state, action):
        # state: [B, seq_len], action: [B, seq_len]
        s_emb = self.state_emb(state)  # [B, seq_len, emb_dim]
        a_emb = self.action_emb(action) # [B, seq_len, emb_dim]
        x = torch.cat([s_emb, a_emb], dim=-1) # [B, seq_len, emb_dim*2]
        out = self.linear(x)
        return out
