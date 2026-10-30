import torch
from abc import ABC, abstractmethod


class BaseKTModel(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, question_seq, concept_seq, response_seq, seq_len):
        """
        :param question_seq: [B, T] 题目id
        :param concept_seq:  [B, T] 知识点id
        :param response_seq: [B, T] 0/1答题标签 float
        :param seq_len:     [B] 每条样本有效长度
        :return probs: [B, T] 每一步答对概率
        """
        pass
