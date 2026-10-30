import torch
import logging
from eduworld.evaluation import compute_dummy_metric
from eduworld.utils import save_metrics, save_checkpoint

logger = logging.getLogger()


class BaseTrainer:
    def __init__(self, model, train_loader, cfg, exp_dir):
        self.model = model
        self.train_loader = train_loader
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.device = torch.device(cfg["reproducibility"]["device"])
        self.model.to(self.device)
        self.epochs = cfg["training"]["epochs"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(cfg["training"]["learning_rate"])
        )

    def train(self):
        for epoch in range(self.epochs):
            logger.info(f"==== Epoch {epoch+1}/{self.epochs} ====")
            # batch现在是dict，不再是(state, action, label)元组
            for batch in self.train_loader:
                # 从字典取出张量并迁移到device
                question_seq = batch["question_seq"].to(self.device)
                concept_seq = batch["concept_seq"].to(self.device)
                response_seq = batch["response_seq"].to(self.device)
                seq_len = batch["seq_len"].to(self.device)

                self.optimizer.zero_grad()
                # 调用B约定的KT模型forward接口
                probs = self.model(question_seq, concept_seq, response_seq, seq_len)

                # response_seq作为标签，binary cross‑entropy损失（dummy可用）
                loss = torch.nn.functional.binary_cross_entropy(probs, response_seq)
                loss.backward()

                self.optimizer.step()
                logger.info(f"step loss: {loss.item():.4f}")

        # 训练结束，评估、保存不变
        metrics = compute_dummy_metric()
        logger.info(f"Eval metrics: {metrics}")
        save_metrics(metrics, f"{self.exp_dir}/metrics/metrics.csv")
        save_checkpoint(self.model, f"{self.exp_dir}/checkpoints/checkpoint.pt")
        logger.info("✅ Smoke test passed")
