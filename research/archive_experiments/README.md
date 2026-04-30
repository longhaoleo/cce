# Archive Experiments

这里保存从当前 Shared 主线中移出的旧实验脚本与旧目录结构，目的是：

- 方便复盘历史实现
- 方便对照过去的命令与模块划分
- 避免这些旧代码继续干扰当前主线

## 说明

- 当前正式主线已经迁到：
  - [runtime/shared/README.md](/root/cce/runtime/shared/README.md)
- 这里的代码默认视为：
  - 归档
  - 只读参考
  - 不保证继续维护

## 当前归档内容

- `scripts/shared/`
  - 旧的 Shared CLI 入口
- `scripts/sdxl_wsae/shared_sae/`
  - 旧的 Shared 实现目录
- `scripts/sdxl_wsae/core/`
  - 旧的通用干预 hook 工具
- `scripts/sdxl_wsae/utils.py`
  - 旧的公共小工具

## 建议

- 新实验优先基于 `runtime/shared/` 继续开发
- 需要找历史逻辑时，再回到这个目录对照
