# Action‑Conditioned 接口说明
目标：支持动力学模型 s_{t+1}=f(s_t, a_t)，支持教学干预推演

## 1. Action如何表示
action代表：题目ID、知识点ID、提示、学习资源、教学干预。
数据集输出每条样本包含 `action_sequence`；
- 数据集输出字段：`state_sequence, action_sequence, label`
- 支持多种action类型，不硬编码死action的含义；后续可扩展action的维度。

## 2. 模型如何接收action
1. 所有KT / WorldModel继承 `BaseWorldModel` 基类
2. forward函数签名强制参数 `forward(state, action)`，禁止只写 `forward(state)`
3. DKT/AKT基线模型暂时可以不使用action做计算，但接口参数必须保留。

## 3. Rollout如何使用action（学生C GRU推演）
1. rollout horizon步数读取自配置文件 `rollout.horizon`
2. 每一步推演输入：当前state + 当前step对应的action
3. 支持两种模式：teacher‑forcing / autoregressive，由配置`rollout.mode`控制
4. rollout循环中迭代传入action_sequence。

## 4. 禁止写死，未来扩展点清单
1. eduworld/data/dataset.py：数据集输出的action字段
2. eduworld/models/base_model.py：BaseWorldModel forward入参
3. rollout推演循环逻辑：action序列传入位置
4. config配置文件的rollout配置块
