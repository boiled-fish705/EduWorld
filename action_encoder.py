"""
Interaction Encoder + action embedding
支持动作集合：Problem, Hint, Review, Explanation, Unknown
独立公共组件，所有模型共享，不耦合业务模型
"""
import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    # 固定动作枚举，顺序不可随意修改，embedding下标一一对应
    ACTION_CANDIDATES = [
        "Problem",
        "Hint",
        "Review",
        "Explanation",
        "Unknown"
    ]

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_action = len(self.ACTION_CANDIDATES)

        # 动作嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=self.num_action,
            embedding_dim=embed_dim
        )

        # 构建字符串→id映射字典，加速查找
        self.action2id = {act: idx for idx, act in enumerate(self.ACTION_CANDIDATES)}

    def _text_to_idx(self, action_text_list):
        """内部工具：字符串列表转为id张量，不在列表内全部映射Unknown"""
        idx_seq = []
        for act in action_text_list:
            if act in self.action2id:
                idx_seq.append(self.action2id[act])
            else:
                idx_seq.append(self.action2id["Unknown"])
        return torch.tensor(idx_seq, dtype=torch.long)

    def forward(self, action_text_list):
        """
        对外前向接口
        :param action_text_list: list[str], 一批动作字符串，例如 ["Problem","Hint"]
        :return: tensor shape [batch_size, embed_dim]
        """
        idx_tensor = self._text_to_idx(action_text_list)
        emb_out = self.embedding(idx_tensor)
        return emb_out


# ----------------可选简易本地自测代码（验证功能，正式运行不会执行）----------------
if __name__ == "__main__":
    encoder = ActionEncoder(embed_dim=16)
    test_actions = ["Problem", "Hint", "Review", "Explanation", "Unknown", "BadActionType"]
    out_emb = encoder(test_actions)
    print(f"输入动作列表：{test_actions}")
    print(f"输出embedding shape：{out_emb.shape}")
    print("ActionEncoder自测通过")
