# 2026-04-28 Shared 主线下 512 分辨率 baseline 画质问题

## 结论

当前 `SharedSAE` 主线里，图像观感差这件事，不能简单归因到：

- SAE 干预挂载
- 概念擦除逻辑
- Shared 特征空间本身

更直接的阶段结论是：

- **`512` 分辨率下的 baseline 生成效果本身就比较糟糕**

也就是说，即使不挂 SAE 干预，只用当前这条 `Shared / SDXL base` 测试链路出图，观感也已经明显偏差。

## 当前配置

Shared checkpoint 当前继承的默认生成配置来自：

- `model_id = stabilityai/stable-diffusion-xl-base-1.0`
- `model_local_dir = /root/autodl-tmp/models/sd-xl-base-1.0-fp16-only`
- `steps = 50`
- `guidance_scale = 8.0`
- `resolution = 512`

对应配置文件：

- `train/output_exp_c_adapter_align/checkpoints/stage3_step_0027400/config.json`

## 当前判断

目前更像是：

- `SDXL base @ 512`
- 无 refiner
- 与 Shared checkpoint 对齐的测试分布

这套组合本身就容易导致：

- 细节粗糙
- 构图不够稳定
- 相比更高分辨率 baseline，视觉质量差很多

所以现在看到的“图像很 low”，至少有一大块原因来自：

- **512 分辨率 baseline 本身质量不高**

而不是：

- Shared 擦除代码一定有 bug

## 研究含义

这件事对后续实验有两个直接影响：

1. 以后评估 Shared 擦除效果时，要把“baseline 本身就不够好看”单独记账  
   不能把所有观感问题都算到擦除头上。

2. `512` 更像是“为了保持 Shared 特征空间/训练分布一致性”的工程选择  
   它不一定是“最佳视觉质量”的选择。

所以后面比较时，需要分清：

- `生成画质`
- `擦除有效性`

这是两件相关但不等价的事情。

## 当前建议

后续如果继续做对照，建议至少保留：

- `512 baseline`
- `1024 baseline`
- `512 + erase`

这样才能区分：

- 是分辨率问题
- 还是擦除问题

一句话总结：

**当前 Shared 主线下，512 分辨率 baseline 画质太差，这本身已经是一个独立问题。**
