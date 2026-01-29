import math
import torch
import torch.nn as nn

from MFFormer.layers.mask import MaskGenerator
from MFFormer.layers.transformer_layers import TransformerBackbone
from MFFormer.layers.features_embedding import (
    StaticEncEmbedding, PatchTokenizer, PatchDepatcher
)

class Model(nn.Module):
    """
    PatchTST-only ablation:
    - Time series -> Patch tokens (per channel) via PatchTokenizer
    - Append STATIC context token
    - Add simple token positional encoding (sinusoidal, no temporal_features, no TFT)
    - transformerBackbone encodes tokens
    - PatchDepatcher reconstructs mu, sigma over [B, T, F]
    - Static head remains StaticEncEmbedding -> StaticDecEmbeddingLSTM

    Expected batch_data_dict keys:
      'batch_x' : [B, T, F]
      'batch_c' : [B, C]
      'batch_time_series_mask_index' : [B, T, F] bool
      'batch_static_mask_index'      : [B, C] bool
      'mode'                         : passed to static decoder
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        d_model = configs.d_model
        d_ffd = configs.d_ffd
        self.embed_dim = d_model

        self.time_series_variables = configs.time_series_variables
        self.static_variables = configs.static_variables
        self.static_variables_category = configs.static_variables_category
        self.static_variables_category_num = [
            len(configs.static_variables_category_dict[x]["class_to_index"])
            for x in configs.static_variables_category
        ]
        self.static_variables_numeric = [
            v for v in self.static_variables if v not in self.static_variables_category
        ]
        self.num_channels = len(self.time_series_variables)

        # Patch tokenization (same as your PatchTokens+TFT model)
        self.use_patches = configs.use_patches
        self.patch_len = configs.patch_len
        self.patch_stride = configs.patch_stride

        self.tokenizer = PatchTokenizer(
            d_model=d_model,
            patch_len=self.patch_len,
            stride=self.patch_stride,
            num_channels=self.num_channels,
            dropout=configs.dropout,
        )

        # Static embedding (same)
        self.static_embedding = StaticEncEmbedding(
            self.static_variables_numeric,
            d_model,
            categorical_features=self.static_variables_category,
            categorical_features_num=self.static_variables_category_num,
            dropout=configs.dropout,
        )

        # Masking (same)
        self.mask_generator = MaskGenerator(
            configs.mask_ratio_time_series,
            mask_ratio_static=configs.mask_ratio_static,
            min_window_size=configs.min_window_size,
            max_window_size=configs.max_window_size,
        )

        # Transformer backbone (same)
        self.encoder = TransformerBackbone(
            d_model, configs.num_enc_layers, d_ffd, configs.num_heads, configs.dropout
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # Depatcher head (same)
        self.depatcher = PatchDepatcher(
            d_model=d_model,
            patch_len=self.patch_len,
            stride=self.patch_stride,
            num_channels=self.num_channels,
        )

        # Static decoder head (same as your PatchTokens+TFT model)
        from MFFormer.layers.features_embedding import StaticDecEmbeddingLSTM
        self.static_projection = StaticDecEmbeddingLSTM(
            self.static_variables_numeric,
            d_model,
            categorical_features=self.static_variables_category,
            categorical_features_num=self.static_variables_category_num,
            dropout=configs.dropout,
            add_input_noise=configs.add_input_noise,
        )

        # Align dims if needed (same)
        self.enc_2_dec_embedding = nn.Linear(d_model, d_model, bias=True)

        self.init_weights()

    def init_weights(self):
        def init_layer(layer):
            if hasattr(layer, "weight") and layer.weight is not None:
                nn.init.uniform_(layer.weight, -self.configs.init_weight, self.configs.init_weight)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.uniform_(layer.bias, -self.configs.init_bias, self.configs.init_bias)

        init_layer(self.tokenizer.patch_proj)
        init_layer(self.depatcher.head)
        init_layer(self.enc_2_dec_embedding)

        for emb in self.static_embedding.numerical_embeddings1.values():
            init_layer(emb)
        for emb in self.static_embedding.numerical_embeddings2.values():
            init_layer(emb)

        nn.init.uniform_(
            self.static_embedding.masked_values,
            -self.configs.init_weight,
            self.configs.init_weight,
        )

    @staticmethod
    def _sinusoidal_positional_encoding(L: int, D: int, device: torch.device) -> torch.Tensor:
        """
        Standard sinusoidal positional encoding for token index positions.
        Returns: [1, L, D]
        """
        pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        div = torch.exp(
            torch.arange(0, D, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / D)
        )  # [D/2]

        pe = torch.zeros(L, D, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, L, D]

    def forward(self, batch_data_dict, is_mask: bool = True):
        batch_x = batch_data_dict["batch_x"]  # [B, T, F]
        batch_c = batch_data_dict["batch_c"]  # [B, C]
        m_ts = batch_data_dict["batch_time_series_mask_index"]  # [B, T, F] bool
        m_st = batch_data_dict["batch_static_mask_index"]       # [B, C] bool
        mode = batch_data_dict.get("mode", None)

        device = batch_x.device
        B, T, F = batch_x.shape
        assert F == self.num_channels, "Input channels mismatch with config time_series_variables"

        # Masking (identical structure to your PatchTokens+TFT model)
        if is_mask:
            if m_ts.numel() == 0:
                _, m_ts = self.mask_generator(batch_x.shape, method="consecutive")
                _, m_st = self.mask_generator(batch_c.shape, method="isolated_point")
            m_ts = m_ts.to(device)
            m_st = m_st.to(device)

            m_missing_ts = torch.isnan(batch_x)
            m_missing_st = torch.isnan(batch_c)
            batch_x = batch_x.masked_fill(m_missing_ts, 0.0)
            batch_c = batch_c.masked_fill(m_missing_st, 0.0)

            m_ts = m_ts | m_missing_ts
            m_st = m_st | m_missing_st
        else:
            m_missing_ts = torch.isnan(batch_x)
            m_missing_st = torch.isnan(batch_c)

        # Patch tokens: [B, F*Np, d_model]
        tokens, Np = self.tokenizer(batch_x)

        # Static token
        enc_c = self.static_embedding(batch_c, feature_order=self.static_variables, masked_index=m_st)  # [B, d_model]
        static_token = enc_c.unsqueeze(1)  # [B, 1, d_model]

        # Concat: [B, L, d_model], where L = F*Np + 1
        enc_x = torch.cat([tokens, static_token], dim=1)

        # Patch-only ablation: add simple token positional encoding (no TFT, no temporal_features)
        L = enc_x.size(1)
        enc_x = enc_x + self._sinusoidal_positional_encoding(L, self.embed_dim, device=device)

        # Encode
        hidden = self.encoder(enc_x)
        hidden = self.encoder_norm(hidden)
        hidden = self.dropout(hidden)
        hidden = self.enc_2_dec_embedding(hidden)

        # Split token/static
        token_states = hidden[:, :-1, :]  # [B, F*Np, d_model]
        static_state = hidden[:, -1, :]   # [B, d_model]

        # Reconstruct mu, sigma over [B, T, F]
        mu, sigma = self.depatcher(token_states, T)

        # Static decoding
        outputs_static, s_start, s_end = self.static_projection(
            static_state, feature_order=self.static_variables, mode=mode
        )

        return {
            "outputs_time_series": mu,
            "outputs_static": outputs_static,

            "outputs_time_series_mu": mu,
            "outputs_time_series_sigma": sigma,

            "masked_time_series_index": m_ts,
            "masked_static_index": m_st,
            "masked_missing_time_series_index": m_missing_ts,
            "masked_missing_static_index": m_missing_st,

            "static_variables_dec_index_start": s_start,
            "static_variables_dec_index_end": s_end,
        }
