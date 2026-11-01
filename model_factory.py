"""
model_factory.py
模型注册工厂 registry，任务3组件
"""
from typing import Dict, Any


class ModelRegistry:
    def __init__(self):
        self._registry: Dict[str, Any] = {}

    def register(self, name: str):
        def decorator(cls):
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str):
        if name not in self._registry:
            raise KeyError(f"模型 {name} 未注册，可用列表：{list(self._registry.keys())}")
        return self._registry[name]

    def list_models(self):
        return list(self._registry.keys())


registry = ModelRegistry()


# 示例基础模型，注册 base_model
@registry.register("base_model")
class BaseModel:
    def __init__(self, encoder):
        self.encoder = encoder
        print("[INFO] BaseModel 初始化完成，绑定action encoder")

    def forward(self, batch):
        emb = self.encoder(batch["action_type"])
        return emb


if __name__ == "__main__":
    print("已注册模型：", registry.list_models())
