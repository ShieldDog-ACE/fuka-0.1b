import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:  # HF trust_remote_code 会把本目录当成顶层包
    from configuration_fuka import FukaConfig
except ImportError:  # 作为 model 包内模块使用
    from .configuration_fuka import FukaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * weight.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class RoPE(nn.Module):
    """位置角表按 (device, dtype) 现算并缓存；不做成 buffer，否则 from_pretrained 的 meta 加载会得到未初始化内存。"""

    def __init__(self, head_dim: int, max_len: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        self.theta = theta
        self._cache: dict = {}

    def _table(self, device, dtype):
        key = (device.type if isinstance(device, torch.device) else device, dtype)
        if key not in self._cache:
            inv = self.theta ** -(torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
            ang = torch.outer(torch.arange(self.max_len, device=device, dtype=torch.float32), inv)
            self._cache[key] = (
                torch.cat([ang.cos(), ang.cos()], dim=-1),
                torch.cat([ang.sin(), ang.sin()], dim=-1),
            )
        return self._cache[key]

    def forward(self, position_ids: torch.Tensor, dtype):
        cos, sin = self._table(position_ids.device, dtype)
        return cos[position_ids].unsqueeze(1).to(dtype), sin[position_ids].unsqueeze(1).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def expand_heads(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    b, kv, t, d = x.shape
    return x[:, :, None].expand(b, kv, repeats, t, d).reshape(b, kv * repeats, t, d)


class Attention(nn.Module):
    def __init__(self, cfg: FukaConfig):
        super().__init__()
        d, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(d, cfg.num_attention_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, cfg.num_key_value_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, cfg.num_key_value_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * hd, d, bias=False)
        self.repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        self.drop = cfg.attention_dropout

    def forward(self, h, cos, sin, mask, past):
        b, t, _ = h.shape
        q = self.q_proj(h).view(b, t, -1, cos.shape[-1]).transpose(1, 2)
        k = self.k_proj(h).view(b, t, -1, cos.shape[-1]).transpose(1, 2)
        v = self.v_proj(h).view(b, t, -1, cos.shape[-1]).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        cache = (k, v)
        if self.repeats > 1:
            k, v = expand_heads(k, self.repeats), expand_heads(v, self.repeats)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.drop if self.training else 0.0,
            is_causal=mask is None and past is None,
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1)), cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: FukaConfig):
        super().__init__()
        d, i = cfg.hidden_size, cfg.intermediate_size
        self.gate = nn.Linear(d, i, bias=False)
        self.up = nn.Linear(d, i, bias=False)
        self.down = nn.Linear(i, d, bias=False)

    def forward(self, h):
        return self.down(F.silu(self.gate(h)) * self.up(h))


class Block(nn.Module):
    def __init__(self, cfg: FukaConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)
        self.drop = cfg.dropout

    def forward(self, h, cos, sin, mask, past):
        a, new_past = self.attn(self.norm1(h), cos, sin, mask, past)
        h = h + a
        m = self.mlp(self.norm2(h))
        if self.drop:
            m = F.dropout(m, self.drop, self.training)
        return h + m, new_past


class FukaPreTrainedModel(PreTrainedModel):
    config_class = FukaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _supports_sdpa = True
    _supports_flash_attn = False

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()


class FukaForCausalLM(FukaPreTrainedModel):
    def __init__(self, config: FukaConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RoPE(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
            self._tied_weights_keys = {"lm_head.weight": "embed.weight"}
        else:
            self._tied_weights_keys = {}
        self.gradient_checkpointing_flag = False
        self.post_init()
        scale = 1.0 / math.sqrt(2 * config.num_hidden_layers)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.endswith(("o_proj.weight", "down.weight")):
                    p.mul_(scale)

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value
        if self.config.tie_word_embeddings:
            self.lm_head.weight = value.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_decoder(self, decoder):
        pass

    def get_decoder(self):
        return self

    def build_mask(self, b, q, kv, padding, device, dtype):
        """仅在需要屏蔽（padding 或带 cache 的分块前向）时构造显式 mask，返回 bool：True=可见。"""
        base = torch.ones(q, kv, dtype=torch.bool, device=device).tril(kv - q)
        if padding is None:
            return base.view(1, 1, q, kv) if b == 1 else base.view(1, 1, q, kv).expand(b, 1, q, kv)
        keep = padding.to(torch.bool).view(b, 1, 1, kv)
        return base.view(1, 1, q, kv) & keep

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        cfg = self.config
        b, t = input_ids.shape
        past = list(past_key_values) if past_key_values else None
        past_len = past[0][0].shape[2] if past else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + t, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(b, t)
        elif position_ids.shape[-1] != t:
            position_ids = position_ids[:, -t:]

        padded = attention_mask is not None and attention_mask.numel() and int(attention_mask.sum()) != b * (past_len + t)
        need_mask = padded or (past is not None and t > 1)
        mask = None
        if need_mask:
            kv = past_len + t
            if padded:
                padding = attention_mask
                if past is not None and padding.shape[-1] > t:
                    padding = padding[:, -kv:]
                if padding.shape[-1] != kv:
                    padding = F.pad(padding, (kv - padding.shape[-1], 0), value=1)
            else:
                padding = torch.ones(b, kv, dtype=torch.long, device=input_ids.device)
            mask_bool = self.build_mask(b, t, kv, padding, input_ids.device, self.embed.weight.dtype)
            mask = torch.where(
                mask_bool,
                torch.zeros((), device=input_ids.device, dtype=self.embed.weight.dtype),
                torch.full((), torch.finfo(self.embed.weight.dtype).min, device=input_ids.device, dtype=self.embed.weight.dtype),
            )

        h = self.embed(input_ids)
        cos, sin = self.rope(position_ids, h.dtype)
        new_past = []
        for i, blk in enumerate(self.blocks):
            p = past[i] if past else None
            if self.gradient_checkpointing_flag and self.training:
                h, pn = checkpoint(blk, h, cos, sin, mask, p, use_reentrant=False)
            else:
                h, pn = blk(h, cos, sin, mask, p)
            new_past.append(pn)
        h = self.final_norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, cfg.vocab_size).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(logits=logits, loss=loss, past_key_values=tuple(new_past) if use_cache else None)

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.gradient_checkpointing_flag = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_flag = False

    @torch.no_grad()
    def sample(self, tokenizer, text: str, max_new: int = 64, temperature: float = 0.0, device="cuda") -> str:
        self.eval()
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        past = None
        for _ in range(max_new):
            out = self(ids[:, -1:] if past else ids, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1]
            nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1) if temperature else logits.argmax(-1, keepdim=True)
            if int(nxt) == tokenizer.eos_token_id:
                break
            ids = torch.cat([ids, nxt], dim=-1)
            past = out.past_key_values
        return tokenizer.decode(ids[0], skip_special_tokens=True)
