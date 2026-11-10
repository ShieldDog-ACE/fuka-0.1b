---
language:
  - en
  - zh
license: other
library_name: transformers
pipeline_tag: text-generation
tags:
  - linux
  - cli
  - tiny-language-model
  - beginner
  - fuka
---

# fuka-0.1b

**v0.0.1** · 109.9M parameters · an ultra-small Linux CLI assistant for beginners

![version](https://img.shields.io/badge/version-v0.0.1-blue)
![params](https://img.shields.io/badge/params-109.9M-green)
![status](https://img.shields.io/badge/status-experimental-orange)

> **English** | [中文](#中文)

> [!WARNING]
> **Experimental release.** Answer accuracy is currently only **30–40%** because of known issues in the data and training pipeline. Generated commands may be wrong or unsafe — always review before running. Not for production use.

## Features

- **Ultra small** — 109.9M parameters, ~420 MiB in fp32. Runs on CPU; no GPU required.
- **Built for Linux beginners** — turns plain-language questions into shell commands, with short explanations.
- **Bilingual** — Chinese and English prompts are both supported.
- **Consistent persona** — speaks as *fuka*, focuses on Linux command-line help rather than general programming.

## Model details

| Item | Value |
|---|---|
| Architecture | Decoder-only Transformer (LLaMA-style) |
| Parameters | 109,926,144 (~109.9M) |
| Layers | 14 |
| Hidden size | 768 |
| Attention heads / KV heads | 12 / 2 (GQA, group size 6) |
| Head dim | 64 |
| FFN | SwiGLU, intermediate size 2048 |
| Norm | RMSNorm (eps 1e-5) |
| Position encoding | RoPE, theta 1,000,000 |
| Context length | 2,048 tokens |
| Vocab | 32,000 (byte-level BPE), tied embeddings |
| Weight dtype | float32 (~420 MiB) |
| Special tokens | `[PAD]` `[BOS]` `[EOS]` `[UNK]` `<\|im_start\|>` `<\|im_sep\|>` `<\|im_end\|>` |

## Repository layout

| Path | Description |
|---|---|
| `linux/` | **Recommended.** Persona SFT + Linux CLI LoRA merged weights |
| `pretrain/` | Base model after pretraining |
| `persona/` | Persona SFT checkpoint (no Linux LoRA) |
| `linux_lora/` | LoRA adapter (r=16, alpha=32, dropout=0.05) for isolation/reuse |
| `fuka_training_data_介绍清单.md` | Dataset card for the training corpus |
| `fuka_training_data_20261110.tar.zst.sha256` | Checksum of the training-data bundle |

> **Weights notice:** `*.safetensors` and the 18.22 GB data archive are not tracked in Git (size limits).
> This repository hosts the model card, code, configs, tokenizer and checksums; v0.0.1 weights are distributed
> separately — contact the author or use your local model directory.

## Quick start

```bash
# Python 3.10+, CPU is enough
pip install torch transformers
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/fuka-0.1b/linux"  # merged persona + Linux CLI weights

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {"role": "user", "content": "How do I list the 10 largest files in the current directory?"}
]
prompt = "".join(
    f"<|im_start|>{m['role']}<|im_sep|>{m['content']}<|im_end|>\n" for m in messages
) + "<|im_start|>assistant<|im_sep|>"

# NOTE: this custom model does not expose .generate(); use its built-in .sample()
print(model.sample(tok, prompt, max_new=48, temperature=0.0, device="cpu"))
```

Prompt format:

```text
<|im_start|>system<|im_sep|>{system prompt}<|im_end|>
<|im_start|>user<|im_sep|>{question}<|im_end|>
<|im_start|>assistant<|im_sep|>
```

## Training data

Trained on a roughly 10B-token bilingual corpus (FineWeb-Edu, FineWeb-Edu-Chinese, Wikipedia en/zh) followed by SFT on
Linux CLI QA (90,570 samples in the final replay set), persona data, and a LoRA pass. See the
[dataset card](fuka_training_data_介绍清单.md) for sources, sizes, licenses, and reproduction steps.

> The 18.22 GB data bundle is not stored in Git. Verify a local copy with:
> `sha256sum -c fuka_training_data_20261110.tar.zst.sha256`

## Evaluation and limitations

- **Accuracy 30–40%** on Linux CLI questions (manual evaluation). The model often produces incomplete or incorrect flags.
- May hallucinate commands, options, or file paths; may suggest destructive operations.
- 2K context window; long documentation does not fit.
- Training data is mixed-license and not fully audited for redistribution.
- v0.0.1 is a **research/experimental** artifact; do not use it in automation or production.

## Version history

- **v0.0.1** (2026-11-10) — first public release: base model, persona SFT, Linux CLI SFT v2 + replay, merged LoRA weights.

## License

The release license is **not finalized** for v0.0.1. Code in this repository is published for evaluation and research;
weights and data remain subject to the upstream licenses listed in the dataset card. Contact the author before commercial use.

## Acknowledgements

Built on the Hugging Face Transformers ecosystem. Training corpora: HuggingFaceFW/fineweb-edu,
opencsg/Fineweb-Edu-Chinese-V2.1, wikimedia/wikipedia, shibing624/alpaca-zh, databricks/databricks-dolly-15k,
tldr-pages.

---

# 中文

**v0.0.1** · 109.9M 参数 · 专为 Linux 新手打造的超小模型

> **中文** | [English](#fuka-01b)

> [!WARNING]
> **实验性版本。** 目前因数据与训练流程存在制作问题，正确率仅有 **30–40%**。生成的命令可能是错的，甚至危险；执行前务必人工确认。请勿用于生产环境。

## 功能

- **超小体积**：109.9M 参数，fp32 约 420 MiB，CPU 即可运行，不需要显卡。
- **面向 Linux 新手**：把自然语言问题转成 shell 命令，并给出简短说明。
- **中英双语**：中文、英文提问都支持。
- **固定人格**：以 fuka（福卡）的身份回答，专注 Linux 命令行，而不是通用编程。

## 模型信息

| 项目 | 数值 |
|---|---|
| 架构 | Decoder-only Transformer（LLaMA 风格） |
| 参数量 | 109,926,144（约 1.1 亿） |
| 层数 | 14 |
| 隐藏维度 | 768 |
| 注意力头 / KV 头 | 12 / 2（GQA，每组 6 头） |
| Head dim | 64 |
| FFN | SwiGLU，中间维度 2048 |
| 归一化 | RMSNorm（eps 1e-5） |
| 位置编码 | RoPE，theta 1,000,000 |
| 上下文长度 | 2,048 tokens |
| 词表 | 32,000（byte-level BPE，嵌入层共享） |
| 权重精度 | float32（约 420 MiB） |
| 特殊 token | `[PAD]` `[BOS]` `[EOS]` `[UNK]` `<\|im_start\|>` `<\|im_sep\|>` `<\|im_end\|>` |

## 仓库结构

| 路径 | 说明 |
|---|---|
| `linux/` | **推荐使用**：人格 SFT + Linux CLI LoRA 合并后的权重 |
| `pretrain/` | 预训练基座 |
| `persona/` | 人格 SFT 权重（未挂 Linux LoRA） |
| `linux_lora/` | LoRA 适配器（r=16，alpha=32，dropout=0.05），便于隔离复用 |
| `fuka_training_data_介绍清单.md` | 训练数据卡（来源、规模、许可证、复现方式） |
| `fuka_training_data_20261110.tar.zst.sha256` | 训练数据包校验和 |

> **权重说明：** `*.safetensors` 与 18.22 GB 数据包不纳入 Git（体积限制）。本仓库只托管模型卡、代码、配置、分词器与校验和；
> v0.0.1 权重单独分发，请联系作者或使用本地模型目录。

## 快速开始

```bash
# Python 3.10+，CPU 即可
pip install torch transformers
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/fuka-0.1b/linux"  # 人格 + Linux CLI 合并权重

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

messages = [{"role": "user", "content": "怎么查看当前目录下最大的 10 个文件？"}]
prompt = "".join(
    f"<|im_start|>{m['role']}<|im_sep|>{m['content']}<|im_end|>\n" for m in messages
) + "<|im_start|>assistant<|im_sep|>"

# 注意：该自定义模型没有 .generate()，请使用内置的 .sample()
print(model.sample(tok, prompt, max_new=48, temperature=0.0, device="cpu"))
```

对话模板：

```text
<|im_start|>system<|im_sep|>{系统提示}<|im_end|>
<|im_start|>user<|im_sep|>{问题}<|im_end|>
<|im_start|>assistant<|im_sep|>
```

## 训练数据

预训练约 100 亿 tokens（FineWeb-Edu、FineWeb-Edu-Chinese、中英文 Wikipedia），随后进行 Linux CLI 问答 SFT
（最终 replay 版 90,570 条）、人格 SFT 与 LoRA 训练。来源、规模、许可证与复现步骤见
[数据卡](fuka_training_data_介绍清单.md)。

> 18.22 GB 数据包不在 Git 仓库中，可用
> `sha256sum -c fuka_training_data_20261110.tar.zst.sha256` 校验本地副本。

## 评测与局限

- Linux CLI 问题上的**正确率仅 30–40%**（人工评测），经常给出不完整或错误的参数。
- 可能编造命令、选项或路径，甚至建议破坏性操作。
- 上下文仅 2K，放不下长文档。
- 训练数据为混合许可证，尚未完成全面合规审计。
- v0.0.1 属于**研究/实验**产物，请勿用于自动化或生产环境。

## 版本记录

- **v0.0.1**（2026-11-10）—— 首次公开发布：基座、人格 SFT、Linux CLI SFT v2 + replay、合并后的 LoRA 权重。

## 许可证

v0.0.1 的发布许可证**尚未最终确定**。仓库内代码仅供评测与研究使用；权重与数据仍受数据卡中列出的上游许可证约束。
商用前请联系作者。

## 致谢

基于 Hugging Face Transformers 生态构建。训练语料来自 HuggingFaceFW/fineweb-edu、opencsg/Fineweb-Edu-Chinese-V2.1、
wikimedia/wikipedia、shibing624/alpaca-zh、databricks/databricks-dolly-15k、tldr-pages。
