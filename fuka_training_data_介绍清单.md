---
language:
  - zh
  - en
license: other
license_name: mixed-see-table
task_categories:
  - text-generation
  - text2text-generation
size_categories:
  - 1B < n < 10B
tags:
  - linux
  - cli
  - pretrain
  - sft
  - fuka
---

# fuka-0.1B 训练数据集 · 开源发布介绍清单

本清单对应压缩包 `fuka_training_data_20261110.tar.zst`（归档 18.22 GB / 约 17.0 GiB；原始 21.38 GB，976 个数据文件，另含 MANIFEST.json 与 README.md），
包含 fuka-0.1B（109.9M 参数，hidden 768 / 14 层 / vocab 32000 / byte-level BPE）从预训练到 SFT、LoRA 全流程实际使用的数据。

## 1. 数据集总览

| # | 数据集 | 角色 | 语言 | 规模 | 上游来源 | 许可证 |
|---|---|---|---|---|---|---|
| 1 | fineweb_edu_en | 预训练·英文通用 | en | 4 文件 / 8.21 GiB / ≈4.0B tokens | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-By 1.0 |
| 2 | chinese_fineweb_edu_v21_45 | 预训练·中文通用 | zh | 900 文件 / 7.83 GiB / ≈4.5B tokens | [opencsg/Fineweb-Edu-Chinese-V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) | Apache-2.0 |
| 3 | wikipedia_en_20231101 | 预训练·英文百科 | en | 8 文件 / 2.36 GiB / ≈1.0B tokens | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | CC BY-SA 4.0 |
| 4 | wikipedia_zh_20231101 | 预训练·中文百科 | zh | 6 文件 / 1.64 GiB / ≈0.48B tokens | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | CC BY-SA 4.0 |
| 5 | alpaca_zh_sft | SFT·中文通用指令 | zh | 1 文件 / 33.5 MiB / 48,818 条 | [shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh) | CC BY 4.0 |
| 6 | dolly_15k_sft | SFT·英文通用指令 | en | 1 文件 / 12.5 MiB / 15,011 条 | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC BY-SA 3.0 |
| 7 | linux_sft.jsonl | SFT v1·Linux CLI | zh/en | 27.6 MiB / 70,955 条 | 自建（tldr / man / docs 生成） | 见 §5 |
| 8 | linux_sft_v2.jsonl | SFT v2·扩充 | zh/en | 31.2 MiB / 75,955 条 | 自建 | 见 §5 |
| 9 | linux_sft_v2_replay.jsonl | **SFT 最终使用版**（含 replay） | zh/en | 34.9 MiB / 90,570 条 | 自建 | 见 §5 |
| 10 | linux_eval.jsonl | 评测集（非训练） | zh/en | 80 KiB / 200 条 | 自建 | 见 §5 |
| 11 | fuka_persona | SFT·人格/说话风格 | zh | 4 文件 / 224 KiB / 395 条 + 历史版本 | 自建 | 自定义，未开放商用 |
| 12 | linux_cli_sft.jsonl | SFT 原始合并源 | zh/en | 16.7 MiB / 44,360 条（zh 15,000 / en 29,360） | 自建，基于 tldr-pages 等 | 见 §5 |
| 13 | qa_archive | 源数据历史版本（追溯用） | zh/en | 42 文件 / 91.9 MiB | 自建 | 见 §5 |
| 14 | docs | QA 生成素材（爬取文档快照） | en/zh | 3 文件 / 97.9 MiB / 47,083 篇 | 爬取，来源各异 | 需逐条复核 |

合计：**约 213.8 亿字节（21.38 GB），976 个数据文件**；预训练语料约 **10B tokens**。

## 2. 包内目录结构

```
fuka_training_data_20261110/
├── MANIFEST.json            # 机器可读清单（大小/文件数/行数/来源/许可证）
├── README.md                # 本文件
├── pretrain_corpus/         # 预训练原始语料（parquet/jsonl，未分词）
│   ├── fineweb_edu_en/
│   ├── chinese_fineweb_edu_v21_45/
│   ├── wikipedia_en_20231101/
│   ├── wikipedia_zh_20231101/
│   ├── alpaca_zh_sft/
│   ├── dolly_15k_sft/
│   ├── manifest.jsonl       # 920 条分片清单
│   └── summary.json         # 采集计划与 token 估算
├── sft_linux/               # 实际送入训练的 SFT 数据（按版本）
│   ├── linux_sft.jsonl
│   ├── linux_sft_v2.jsonl
│   ├── linux_sft_v2_replay.jsonl   # 最终版
│   └── linux_eval.jsonl
├── sft_persona/             # fuka 人格/口吻数据 + _archive 历史版本
└── source/                  # 原始素材
    ├── linux_cli_sft.jsonl  # 合并后的 44,360 条源数据
    ├── qa_archive/          # 中/英/生成/人工/KB 各分支历史版本
    └── docs/                # 爬取文档快照（2026-10-30 ~ 11-01）
```

`pretrain_corpus/*.bin` 与 `.output/cache/*.npy` 属于可再生的分词/打包产物（约 16 GiB），未重复收录；
复现方式见 §4。

## 3. 训练数据配方

| 阶段 | 数据 | 说明 |
|---|---|---|
| 预训练 | fineweb-edu(en) + Fineweb-Edu-Chinese-V2.1(zh) + Wikipedia(en/zh) | 约 10B tokens 混合，中文占比约 50%，打包为 920 个分片 |
| 通用 SFT | alpaca-zh + dolly-15k | 中英通用指令跟随 |
| 领域 SFT | linux_cli_sft → linux_sft → v2 → v2_replay | Linux CLI 中英问答；replay 版混入旧数据缓解遗忘，为最终采用版本 |
| 人格 SFT | fuka_persona | 自我介绍、语气、边界（不写代码、专注 Linux 命令行） |
| LoRA | linux_sft_v2_replay | r=16, alpha=32, dropout=0.05，target: q/k/v/o/gate/up/down；已合并进发布权重 |

## 4. 复现方式

```bash
# 1) 解包
tar -I zstd -xf fuka_training_data_20261110.tar.zst

# 2) 预训练语料 → 分词分片（在原始工程目录执行）
python data/build_pretrain_bins.py --corpus <解包后的 pretrain_corpus> --out .output/pretrain

# 3) SFT / LoRA
python train/run.py --stage sft   --data sft_linux/linux_sft_v2_replay.jsonl
python train/run.py --stage lora  --data sft_linux/linux_sft_v2_replay.jsonl --r 16 --alpha 32
```

SFT/JSONL 统一字段：`id, lang, kind, topic, command, source, messages[{role, content}]`；
persona 数据字段：`id, lang, kind, messages[]`。

## 5. 许可证与合规提示

可直接再分发（需保留署名与许可证）：

- fineweb-edu：ODC-By 1.0
- Fineweb-Edu-Chinese-V2.1：Apache-2.0
- Wikipedia 20231101：CC BY-SA 4.0（衍生数据需同协议共享）
- alpaca-zh：CC BY 4.0
- databricks-dolly-15k：CC BY-SA 3.0

需在公开发布前复核：

- **自建 Linux SFT / persona / qa_archive**：内容源自 tldr-pages、man pages、在线文档与模型生成，混合许可；建议逐条保留 `source` 字段，发布时剔除来源不明或含敏感信息的样本。
- **docs/**：爬取快照，各页面版权归原作者，许可证不统一；若上架公开平台，建议不随包发布，或仅发布文档 URL 清单。
- tldr-pages 内容通常为 CC BY 4.0，引用时请注明 `https://github.com/tldr-pages/tldr`。
- persona 数据含角色设定，发布前确认无个人隐私信息。

## 6. 建议的平台填写信息

- 数据集名称：`fuka-0.1b-training-data`
- 任务类型：text-generation / text2text-generation
- 语言：zh, en
- 规模：1B–10B tokens（预训练）+ ~10 万条 SFT
- 标签：linux, cli, instruction-tuning, pretrain, sft, lora
- 许可：`other`（混合许可，详见 §5），或按子集分别建仓
- 推荐拆分（更利于合规与复用）：
  1. `fuka-pretrain-corpus`（4 个公开语料，纯聚合）
  2. `fuka-linux-cli-sft`（自建，复核后发布）
  3. `fuka-persona-sft`（可选，视隐私复核结果）

## 7. 引用

```bibtex
@misc{fuka2026trainingdata,
  title  = {fuka-0.1B Training Data Bundle},
  author = {fuka project},
  year   = {2026},
  note   = {Pretrain corpus + Linux CLI SFT + persona data},
  url    = {<发布后填写>}
}
```

上游数据集请同时引用其原始仓库（FineWeb-Edu、FineWeb-Edu-Chinese、Wikipedia、alpaca-zh、databricks-dolly-15k、tldr-pages）。
