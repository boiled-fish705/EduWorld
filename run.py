import argparse
from dataset import get_dataloader
from action_encoder import ActionEncoder
from model_factory import registry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_dummy", action="store_true")
    args = parser.parse_args()

    print(f"[INFO] 加载数据集，use_dummy = {args.use_dummy}")
    train_loader = get_dataloader("", batch_size=4, use_dummy=args.use_dummy)

    print("[INFO] 初始化Action Encoder")
    encoder = ActionEncoder()

    model_name = "base_model"
    print(f"[INFO] 获取注册模型：{model_name}")
    print(f"[INFO] 当前已注册模型列表: {registry.list_models()}")
    model_cls = registry.get(model_name)
    model = model_cls(encoder=encoder)

    print("[INFO] 组件全部加载完成，可以开始训练/推理")

    # ==========新增：拿一个batch跑一次前向，验证全链路==========
    for batch in train_loader:
        emb_out = model.forward(batch)
        print(f"[INFO] 前向推理成功，输出shape: {emb_out.shape}")
        break

if __name__ == "__main__":
    main()
