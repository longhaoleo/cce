# SAE Data Layout

统一的 SAE 强绑定产物目录：

```text
sae_data/<sae_tag>/
  concept-dig/
  concept-dig-freq/
  blacklist/
  feature-freq/
```

含义：

- `concept-dig/`
  - 概念定位输出
- `concept-dig-freq/`
  - blacklist 第二遍构建时导出的排序表
- `blacklist/`
  - 正式给 locator / erase / batch 读取的 `feature_blacklist.txt`
- `feature-freq/`
  - prompt-conditioned 基础统计

当前已迁移：

- `sae_x8_time`
- `sae_x8_time_decorr03`

补充约定：

- `sae_x8_time_decorr03` 当前根级默认 `blacklist/` 和 `concept-dig-freq/` 对应的是 `ar95_all`
- 其他变体仍然保留在：
  - `blacklist/ar90_all/`
  - `blacklist/q99_50/`
  - `blacklist/q99_50_initial/`
  - `concept-dig-freq/ar90_all/`
  - `concept-dig-freq/q99_50/`
  - `concept-dig-freq/q99_50_initial/`

推荐命令一律带：

```bash
--sae_root sae_data/<sae_tag>
```
