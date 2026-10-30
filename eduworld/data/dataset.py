import torch
from torch.utils.data import Dataset


class BaseKTDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_samples = 8

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        max_seq_len = self.cfg["dataset"]["max_seq_len"]
        question_seq = torch.randint(1, 100, (max_seq_len,), dtype=torch.long)
        concept_seq = torch.randint(1, 20, (max_seq_len,), dtype=torch.long)
        response_seq = torch.randint(0, 2, (max_seq_len,), dtype=torch.float32)
        real_len = torch.randint(5, max_seq_len, (), dtype=torch.long)

        sample = {
            "question_seq": question_seq,
            "concept_seq": concept_seq,
            "response_seq": response_seq,
            "seq_len": real_len
        }
        return sample
