from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    model_type = "my_model"
    def __init__(self,
        vocab_size=50304,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        # initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        flash_attn=False,
        # tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=1.0,
        # use_sliding_window=False,
        # sliding_window=4096,
        # max_window_layers=28,
        attention_dropout=0.0,
        num_additional_preds=2,
        mtp_lambda_weight=1.0,
        **kwargs,):
        
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        # self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.flash_attn = flash_attn
        # self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # self.use_sliding_window = use_sliding_window
        # self.sliding_window = sliding_window
        # self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.num_additional_preds = num_additional_preds
        self.mtp_lambda_weight = mtp_lambda_weight