\# W02\_D\_encoder\_action\_v0.1

\## 版本信息

版本：v0.1

模块：Interaction Encoder + action embedding

开发目标：实现公共action编码组件，独立于模型，对外统一接口



\## 1. Action类别定义

action类别一共5类：

\- Problem

\- Hint

\- Review

\- Explanation

\- Unknown



\## 2. 模块定位

本模块是项目公共组件，不耦合在任何单个模型内部，由model\_factory、dataloader调用。

输入：原始action文本/action标签

输出：action embedding向量



\## 3. 对外接口

类名：ActionEncoder

文件路径：action\_encoder.py

核心方法：

1\. \_\_init\_\_()：初始化编码器、词表、embedding层

2\. forward(action\_text\_list) → embedding tensor

&#x20;   - action\_text\_list：字符串列表，为上面5种类别之一

&#x20;   - 返回：可直接送入模型的embedding向量



\## 4. 调用示例

```python

from action\_encoder import ActionEncoder



encoder = ActionEncoder()

emb = encoder(\["Problem","Hint"])



