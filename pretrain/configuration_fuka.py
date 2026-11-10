from transformers import PretrainedConfig


class FukaConfig(PretrainedConfig):
    model_type = "fuka"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=14,
        num_attention_heads=12,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=2048,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        dropout=0.0,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        kwargs["tie_word_embeddings"] = tie_word_embeddings
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def num_attention_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def estimated_params(self) -> int:
        d, h, inter = self.hidden_size, self.head_dim, self.intermediate_size
        kv = self.num_key_value_heads * h
        emb = self.vocab_size * d * (2 if not self.tie_word_embeddings else 1)
        attn = d * d + 2 * d * kv + d * d
        mlp = 3 * d * inter
        per_layer = attn + mlp + 2 * d
        return emb + self.num_hidden_layers * per_layer + d
