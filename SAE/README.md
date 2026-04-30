# SAE Package

这个目录现在代表**当前项目正在使用的 SAE 实现**，也就是 SharedSAE。

它不再保存旧的单块 SAE / WebDataset 训练器，而是作为一个更稳定的模型包入口，统一暴露：

- 模型结构
- checkpoint 读写
- 时间/空间编码
- block 归一化
- 训练配置

## 文件

- `sae.py`
  - SharedSAE 模型与训练相关辅助函数
- `checkpoint.py`
  - checkpoint 保存/加载接口
- `encoding.py`
  - 时间/空间编码与坐标工具
- `normalization.py`
  - block 归一化系数统计与应用
- `config.py`
  - Shared 训练配置对象

## 说明

- 这里现在不再是薄薄的一层转发，而是当前 SharedSAE 的实体实现目录
- `sae.py / checkpoint.py / encoding.py / normalization.py / config.py` 都可以直接修改
- 这样做的目的是把“你的 SAE 是什么”单独收成一个稳定模块目录
- 后面如果你继续拆模型内部结构，可以优先沿着这个目录继续展开
