# EXP04 — 分布式程序状态恢复（Distributed Procedural-State Restoration）

EXP03 表明单 token 精确互换不充分。这尚未证明分布式电路：单 token 供体修补可能只是创建了不一致的供体/接收者混合状态。

EXP04 检验跨越多个共享后缀 token 和/或多个层的供体激活**一致集合**能否转移供体技能的下一次动作偏好。

## 主检验

H19 处的 token 分布：
- `last1_H19`
- `common_H19`

精确最长公共 token 后缀上的层分布：
- `common_H19_H24`
- `common_H19_H28`  **主检验**
- `common_H21_H28`
- `common_H15_H28`

`H19` 是 Hugging Face 隐藏状态索引 19，即 decoder block 18 输出。

## 主供体

同任务 + 同措辞 + 反规定状态。

## 控制条件

在主检验 `common_H19_H28` 配置下：
- 自修补（self patch）；
- 同状态 + 反措辞；
- 反状态 + 反措辞。

最后一个控制检验转移的因果状态能否跨技能表面措辞泛化。

## 主指标

`margin = mean_logP(src action) - mean_logP(test action)`

当修补后的接收者移向供体技能的规定状态时，有符号转移效应为正。

## 运行方式

在仓库根目录执行：

```bash
python experiments/exp04_distributed_restoration/run.py
```

脚本完全离线，自动检测 `./models` 下的本地模型。

## 输出

`outputs/exp04_distributed_restoration/`

关键文件：
- `summary.json`
- `config_effects.csv`
- `task_effects_primary.csv`
- `common_suffix_lengths.csv`
- `distributed_restoration.png`
- `run_manifest.json`